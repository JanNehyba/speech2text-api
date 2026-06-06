import asyncio
import logging
import os
import shutil
import tempfile
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from speech2text_api.settings import settings
from speech2text_api.whisper import Whisper

# Optional Sentry middleware. Upstream pinned ``sentry-asgi==0.2.0`` (2020),
# which doesn't work with modern starlette. We use ``sentry-sdk``'s built-in
# ASGI middleware when available and a Sentry DSN is configured; otherwise
# we just skip it.
try:
    from sentry_sdk.integrations.asgi import SentryAsgiMiddleware  # type: ignore
except ImportError:  # pragma: no cover - sentry is optional
    SentryAsgiMiddleware = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class HelloResponse(BaseModel):
    message: str


class TranscriptSegment(BaseModel):
    """One coherent stretch of speech with start/end timestamps in seconds."""
    start: float
    end: float
    text: str


class TranscriptResponse(BaseModel):
    """Response model.

    ``segments`` is omitted (None) for legacy callers that don't ask for
    timestamps, preserving wire compatibility with the original upstream.
    """
    transcript: str
    segments: Optional[List[TranscriptSegment]] = None


class DiarizationTurn(BaseModel):
    """One speaker turn with start/end timestamps in seconds."""
    start: float
    end: float
    speaker_label: str


class DiarizationSpeakerEmbedding(BaseModel):
    """Mean embedding vector for one detected speaker, weighted by speech duration.

    Used by QualReAI as a long-lived "voice profile" so the same person can
    be auto-suggested when they appear in subsequent recordings of the same
    project.
    """
    vector: List[float]
    seconds_observed: float


class DiarizationSegmentEmbedding(BaseModel):
    """Per-turn embedding, returned only when ``return_segment_embeddings=true``.

    Persisted on Segment rows so QualReAI's merge/split operations can
    recompute speaker profiles exactly without re-running pyannote.
    """
    start: float
    end: float
    speaker_label: str
    vector: List[float]


class DiarizationResponse(BaseModel):
    turns: List[DiarizationTurn]
    num_speakers_detected: int
    diarization_model: str
    speaker_embeddings: Dict[str, DiarizationSpeakerEmbedding]
    segment_embeddings: Optional[List[DiarizationSegmentEmbedding]] = None


def get_app() -> FastAPI:
    app = FastAPI(
        title=settings.title,
        description=settings.description,
        debug=settings.debug_api,
        openapi_url=settings.openapi_route,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"])
    if SentryAsgiMiddleware is not None and settings.sentry_dsn:
        app.add_middleware(SentryAsgiMiddleware)
    return app


app = get_app()

INDEX_HTML = """
<h1>Speech2Text API</h1>
<p>
<a href="docs">API documentation</a>
</p>
<p>
Add <code>?timestamps=true</code> to <code>/transcribe/{lang}/</code> to get
segment-level start/end times for each phrase.
</p>
<p>
POST <code>/diarize/</code> for speaker diarization (pyannote.audio 4.x, community-1).
</p>
""".lstrip()


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/healthz", include_in_schema=False)
def healthz() -> Dict[str, str]:
    """Liveness probe — does not load models, just confirms the process is up."""
    return {"status": "ok"}


# Single Whisper instance — model is loaded once at process startup.
speech2text_wrapper = Whisper(settings.model_id)


async def create_tmp_file(file: UploadFile, output_fname: str = "input.wav") -> str:
    tmp_dir = tempfile.mkdtemp()
    # librosa picks the decoder based on the file suffix, so we keep .wav
    # as a generic placeholder; ffmpeg/librosa will still decode mp3/m4a/etc.
    tmp_fname = os.path.join(tmp_dir, output_fname)
    with open(tmp_fname, "wb") as wav_tmp_f:
        wav_tmp_f.write(file.file.read())
    return tmp_fname


@app.post("/transcribe/", response_model=TranscriptResponse)
async def transcribe_default(
    file: UploadFile,
    timestamps: bool = False,
    beam_size: Optional[int] = None,
    vad_filter: Optional[bool] = None,
) -> TranscriptResponse:
    tmp_fname = await create_tmp_file(file)
    result = speech2text_wrapper.transcribe_file(
        tmp_fname,
        return_timestamps=timestamps,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
    return TranscriptResponse(**result)


@app.post("/transcribe/{lang}/", response_model=TranscriptResponse)
async def transcribe_chosen_lang(
    lang: str,
    file: UploadFile,
    timestamps: bool = False,
    beam_size: Optional[int] = None,
    vad_filter: Optional[bool] = None,
) -> TranscriptResponse:
    tmp_fname = await create_tmp_file(file)
    result = speech2text_wrapper.transcribe_file(
        tmp_fname,
        lang=lang,
        return_timestamps=timestamps,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
    return TranscriptResponse(**result)


@app.post("/diarize/", response_model=DiarizationResponse)
async def diarize(
    file: UploadFile,
    num_speakers: int = 2,
    min_duration: float = 0.5,
    return_segment_embeddings: bool = False,
    exclusive_speaker_mode: bool = False,
) -> DiarizationResponse:
    """Run pyannote.audio speaker diarization on an uploaded audio file.

    Default model is ``pyannote/speaker-diarization-community-1`` (4.0
    family, VBx clustering). Override via the ``PYANNOTE_DIARIZATION_MODEL``
    env var on the pod.

    Query params:
        num_speakers: Exact speaker count hint (default 2 for 1-on-1
            interviews). Forces the clustering step to that many clusters
            — the biggest single accuracy win for the 1-on-1 case.
        min_duration: Discard turns shorter than this many seconds (default
            0.5). Filters out sub-100ms silence/cough artefacts.
        return_segment_embeddings: When True, also return one embedding
            vector per turn. Used by QualReAI's merge/split flow in
            Phase 2. Off by default — embedding crops cost ~10-100 ms
            per turn on CPU and the F1 path doesn't need them.
        exclusive_speaker_mode: When True, return the pyannote 4.x
            *exclusive* diarization (one speaker per moment, no
            overlap). Default False keeps the standard with-overlap
            output that matches pyannote 3.x behaviour. No effect on
            a 3.x pipeline.

    Concurrency: capped to one in-flight diarization per process. The
    second concurrent caller waits on a semaphore. Diarization on the
    MU-sized CPU pod runs ~0.3-0.5× realtime, so a 45-min audio file
    holds the semaphore for ~20-45 minutes — clients must use a matching
    HTTP timeout.

    Returns 503 if pyannote can't initialise (typically a missing
    HF_TOKEN secret or unaccepted model EULA on huggingface.co).
    """
    from speech2text_api.diarize import (  # noqa: PLC0415 - lazy import for fast startup
        diarize_audio_file,
        get_concurrency_semaphore,
    )

    tmp_fname = await create_tmp_file(file, output_fname="diarize_input.wav")
    tmp_dir = os.path.dirname(tmp_fname)

    sem = get_concurrency_semaphore()
    try:
        async with sem:
            # Run the synchronous pyannote inference in a worker thread so
            # the event loop stays responsive to other requests / probes.
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: diarize_audio_file(
                        tmp_fname,
                        num_speakers=num_speakers,
                        min_duration=min_duration,
                        return_segment_embeddings=return_segment_embeddings,
                        exclusive_speaker_mode=exclusive_speaker_mode,
                    ),
                )
            except RuntimeError as exc:
                logger.exception("Diarization pipeline initialisation failed")
                raise HTTPException(
                    status_code=503, detail=str(exc)
                ) from exc
            except Exception as exc:  # noqa: BLE001
                logger.exception("Diarization failed")
                raise HTTPException(
                    status_code=500, detail=f"Diarization failed: {exc}"
                ) from exc
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass

    return DiarizationResponse(
        turns=[
            DiarizationTurn(
                start=t.start, end=t.end, speaker_label=t.speaker_label
            )
            for t in result.turns
        ],
        num_speakers_detected=result.num_speakers_detected,
        diarization_model=result.diarization_model,
        speaker_embeddings={
            label: DiarizationSpeakerEmbedding(
                vector=stats.mean_embedding,
                seconds_observed=stats.seconds_observed,
            )
            for label, stats in result.speaker_embeddings.items()
        },
        segment_embeddings=(
            [
                DiarizationSegmentEmbedding(
                    start=s["start"],
                    end=s["end"],
                    speaker_label=s["speaker_label"],
                    vector=s["vector"],
                )
                for s in result.segment_embeddings
            ]
            if result.segment_embeddings is not None
            else None
        ),
    )
