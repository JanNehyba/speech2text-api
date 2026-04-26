import logging
import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, UploadFile
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
""".lstrip()


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


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
