"""Speaker diarization via pyannote.audio 4.x (community-1 pipeline).

CPU-only inference. Models are loaded lazily on the first request so the
container can pass liveness probes during start-up without paying the
download cost up-front (the Dockerfile snapshot-downloads weights for both
3.1 and community-1 at build time, so first-request loading is just a
torch ``state_dict`` load from disk, sub-second). Which pipeline gets
loaded is controlled by the ``PYANNOTE_DIARIZATION_MODEL`` env var; default
is ``pyannote/speaker-diarization-community-1`` (4.0 family with VBx
clustering, deep-research 2026-06-05 finding).

Concurrency: pyannote peaks at ~2-3 GiB of working memory and saturates
all CPU cores. The MU pod runs faster-whisper alongside; running two
diarizations in parallel would OOM the pod (12 GiB limit) and starve the
Whisper path. ``get_concurrency_semaphore()`` returns a single-permit
asyncio semaphore that the HTTP layer must hold around the synchronous
diarize call.

**MP3 pre-conversion (issue #1752 workaround)**: Pyannote's internal
``Audio.crop()`` uses ``torchaudio.load()`` which is not sample-accurate
for MP3 files — the trailing 10-second embedding chunk comes back as
e.g. 146456 samples instead of the expected 160000, and ``torch.vstack``
crashes in ``SpeakerDiarization.get_embeddings``. The maintainer marked
this ``wontfix`` (pyannote relies on torchaudio; MP3 backend behaviour
is out of scope). To dodge the bug we pre-decode every incoming audio
file to a clean 16 kHz mono PCM_16 WAV in ``/tmp`` via librosa +
soundfile, then hand that WAV path to pyannote. librosa already lives
in the image as a faster-whisper transitive dep, so no new wheels.

For the cross-transcript voice-ID feature in QualReAI we return two kinds
of embeddings:

* ``speaker_embeddings`` — one mean vector per detected speaker, weighted
  by total speech duration. Used as the long-lived "voice profile". In
  pyannote 4.x these come back on ``DiarizeOutput.speaker_embeddings``
  (np.ndarray of shape ``(num_speakers, dim)``, ordered to match
  ``output.speaker_diarization.labels()``). No more ``return_embeddings=True``
  kwarg — they're always returned unless ``legacy=True`` was set at
  pipeline construction.
* ``segment_embeddings`` (opt-in via ``return_segment_embeddings=True``)
  — one vector per turn, persisted on ``Segment`` rows so merge/split
  operations can recompute profiles exactly without re-running pyannote.
  All vectors come from the *same* embedding model the diarization
  pipeline uses internally, so cosine similarities are directly
  comparable across both shapes.

**Exclusive speaker mode** (pyannote 4.x feature): the pipeline always
computes both a regular diarization (with overlap regions assigned to
multiple speakers) and an *exclusive* diarization (every moment belongs
to at most one speaker). Toggling via ``exclusive_speaker_mode=True``
on ``diarize_audio_file`` picks the exclusive annotation as the source
of truth for the returned turns — useful when the downstream consumer
(transcript alignment, coding UI) can't represent overlap and would
otherwise produce ``None``/cross-talk artefacts. Speaker labels and
embedding rows are identical across both modes; only the turn boundaries
differ.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Inference, Model, Pipeline
from pyannote.core import Segment

logger = logging.getLogger(__name__)

DIARIZATION_MODEL = os.environ.get(
    "PYANNOTE_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
)
# Fallback embedding model used only if ``pipeline.embedding`` / ``_embedding``
# isn't exposed by the pyannote version we end up with. Matches the model
# community-1 uses internally (a WeSpeaker variant) so per-turn embeddings
# stay comparable to the pipeline output. wespeaker-voxceleb-resnet34-LM
# also works as a 3.1-compatible fallback.
FALLBACK_EMBEDDING_MODEL = os.environ.get(
    "PYANNOTE_EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM"
)

_init_lock = threading.Lock()
_pipeline: Optional[Pipeline] = None
_embedding_inference: Optional[Inference] = None

# Single-permit semaphore for the HTTP layer. Created lazily on first use so
# it binds to the current running event loop instead of being created at
# import time when no loop exists yet.
_semaphore: Optional[asyncio.Semaphore] = None
_semaphore_lock = threading.Lock()


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker_label: str


@dataclass(frozen=True)
class SpeakerStats:
    label: str
    seconds_observed: float
    mean_embedding: list[float]


@dataclass(frozen=True)
class DiarizationResult:
    turns: list[SpeakerTurn]
    num_speakers_detected: int
    diarization_model: str
    speaker_embeddings: dict[str, SpeakerStats]
    segment_embeddings: Optional[list[dict]]


def get_concurrency_semaphore() -> asyncio.Semaphore:
    """Return a single-permit semaphore bound to the running event loop."""
    global _semaphore
    if _semaphore is not None:
        return _semaphore
    with _semaphore_lock:
        if _semaphore is None:
            _semaphore = asyncio.Semaphore(1)
    return _semaphore


# Env-var escape hatch: set PYANNOTE_PRECONVERT_TO_WAV=false to disable the
# librosa pre-decode and pass the raw audio path straight to pyannote. Useful
# if a future audio format breaks librosa but works for torchaudio, or for
# a quick rollback if the WAV write step itself misbehaves.
_PRECONVERT_TO_WAV = os.environ.get(
    "PYANNOTE_PRECONVERT_TO_WAV", "true"
).strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _ensure_pcm_wav(audio_path: str) -> Iterator[str]:
    """Yield a path to a 16 kHz mono PCM_16 WAV view of ``audio_path``.

    For pre-existing WAV inputs we still re-encode through librosa+soundfile
    so the file is guaranteed sample-aligned at 16 kHz mono — torchaudio's
    WAV reader is sample-accurate but the upstream file might be 44.1/48 kHz
    or stereo, which pyannote would have to resample on-the-fly anyway.
    The decode-write round-trip is ~500 ms for a 45-minute file; acceptable
    next to ~25 minutes of pyannote inference that follows.

    Cleanup is best-effort: any OSError on unlink is swallowed (e.g. file
    already gone if /tmp was wiped by an external process).
    """
    if not _PRECONVERT_TO_WAV:
        yield audio_path
        return

    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="diarize_pcm_")
    os.close(fd)
    try:
        logger.info("Pre-decoding %s -> %s (16 kHz mono PCM)", audio_path, wav_path)
        # librosa.load handles MP3/M4A/MP4/WAV uniformly via soundfile+audioread+ffmpeg,
        # always returns a clean np.ndarray[float32] with exactly sr*duration samples
        # at the target rate. No MP3 frame-padding artefacts.
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        if waveform.size == 0:
            raise RuntimeError(
                f"librosa.load produced empty waveform for {audio_path!r}"
            )
        # PCM_16 keeps the file ~8x smaller than float32 (45 min audio = ~85 MB)
        # while staying lossless for what pyannote actually consumes.
        sf.write(wav_path, waveform, 16000, subtype="PCM_16")
        yield wav_path
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _load_pipeline() -> Pipeline:
    """Lazily load the pyannote SpeakerDiarization pipeline on CPU."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _init_lock:
        if _pipeline is None:
            hf_token = os.environ.get("HF_TOKEN") or None
            logger.info(
                "Loading pyannote pipeline %s (CPU)", DIARIZATION_MODEL
            )
            # pyannote 4.x renamed the HF auth kwarg from
            # use_auth_token -> token (matching huggingface_hub). The 3.x
            # name was removed outright and now raises TypeError.
            pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                token=hf_token,
            )
            if pipeline is None:
                raise RuntimeError(
                    f"Pipeline.from_pretrained returned None for "
                    f"{DIARIZATION_MODEL!r}. Did you accept the model EULA on "
                    f"huggingface.co and provide HF_TOKEN?"
                )
            pipeline.to(torch.device("cpu"))
            _pipeline = pipeline
    return _pipeline


def _get_embedding_inference(pipeline: Pipeline) -> Inference:
    """Get Inference for per-turn embedding crops, compatible with pipeline output.

    Prefers the pipeline's own embedding submodel so per-turn vectors
    occupy the same space as the mean-per-speaker vectors returned on
    ``DiarizeOutput.speaker_embeddings``. Falls back to a canonical
    pyannote model.

    pyannote 4.x stores embeddings as ``pipeline._embedding`` (a
    ``PretrainedSpeakerEmbedding`` wrapper); 3.x exposed ``pipeline.embedding``
    as a raw ``Model``. We try both, and if neither yields an ``Inference``
    we can crop with, we load a standalone fallback Model and wrap it.
    """
    global _embedding_inference
    if _embedding_inference is not None:
        return _embedding_inference
    with _init_lock:
        if _embedding_inference is None:
            # pyannote 3.x → `.embedding` (raw Model); 4.x → `._embedding`
            # (PretrainedSpeakerEmbedding) and `.embedding` is the dict spec
            # used at construction. We probe both, but require something
            # that is either an Inference or wrappable in one (a Model).
            emb_candidate = (
                getattr(pipeline, "_embedding", None)
                or getattr(pipeline, "embedding", None)
            )
            wrappable = isinstance(emb_candidate, (Inference, Model))

            if not wrappable:
                hf_token = os.environ.get("HF_TOKEN") or None
                logger.info(
                    "Pipeline did not expose a wrappable embedding model "
                    "(got %s); loading fallback %s",
                    type(emb_candidate).__name__,
                    FALLBACK_EMBEDDING_MODEL,
                )
                emb_candidate = Model.from_pretrained(
                    FALLBACK_EMBEDDING_MODEL,
                    token=hf_token,
                )
                emb_candidate.to(torch.device("cpu"))

            if isinstance(emb_candidate, Inference):
                _embedding_inference = emb_candidate
            else:
                _embedding_inference = Inference(emb_candidate, window="whole")
    return _embedding_inference


def diarize_audio_file(
    audio_path: str,
    num_speakers: int = 2,
    min_duration: float = 0.5,
    return_segment_embeddings: bool = False,
    exclusive_speaker_mode: bool = False,
) -> DiarizationResult:
    """Run diarization on an audio file.

    Args:
        audio_path: Path to a decodable audio file. Resampling to 16 kHz
            mono is handled by pyannote / torchaudio internally.
        num_speakers: Exact speaker count hint. Forces the clustering step
            to that many clusters — the single biggest accuracy win for the
            1-on-1 interview case in QualReAI (default 2).
        min_duration: Discard turns shorter than this many seconds.
            pyannote sometimes emits sub-100ms artefacts on silence.
        return_segment_embeddings: When True, also return a per-turn
            embedding vector. Cost: one model crop per turn (~10-100 ms
            each on CPU), so disable for the F1 path that only needs
            mean-per-speaker.
        exclusive_speaker_mode: When True (pyannote 4.x), return the
            *exclusive* diarization annotation where every moment belongs
            to at most one speaker (no cross-talk overlap). Useful when
            the downstream consumer can't represent overlapping turns and
            would otherwise produce ``None``/cross-talk gaps. Default False
            keeps the standard with-overlap output that matches 3.x
            behaviour. Has no effect on a 3.x pipeline (which doesn't
            produce an exclusive annotation) — we silently fall back to
            the regular annotation in that case.
    """
    pipeline = _load_pipeline()

    # Pre-convert any incoming audio (MP3, M4A, MP4, WAV, ...) to a clean
    # 16 kHz mono PCM WAV in /tmp. This dodges pyannote-audio issue #1752
    # (torchaudio MP3 seek not sample-accurate -> torch.vstack crash in
    # SpeakerDiarization.get_embeddings). The bug carries over to pyannote
    # 4.x identically (upstream wontfix — torchaudio is still the loader).
    # WAV produced by soundfile has exact sample alignment, so Audio.crop
    # returns full 10s = 160000-sample chunks every time.
    with _ensure_pcm_wav(audio_path) as wav_path:
        # pyannote 4.x: pipeline(audio, num_speakers=N) returns a
        # DiarizeOutput dataclass with .speaker_diarization,
        # .exclusive_speaker_diarization, and .speaker_embeddings. The
        # old `return_embeddings=True` kwarg is gone — embeddings are
        # always returned unless the pipeline was constructed with
        # `legacy=True`.
        # pyannote 3.x: same call without that kwarg returns just an
        # Annotation; we detect both shapes below.
        raw_output = pipeline(wav_path, num_speakers=num_speakers)

        # Normalise to (annotation, mean_embeddings) across pyannote 3.x and 4.x.
        if hasattr(raw_output, "speaker_diarization"):
            # pyannote 4.x DiarizeOutput
            mean_embeddings = getattr(raw_output, "speaker_embeddings", None)
            if exclusive_speaker_mode and hasattr(
                raw_output, "exclusive_speaker_diarization"
            ):
                annotation = raw_output.exclusive_speaker_diarization
            else:
                annotation = raw_output.speaker_diarization
        elif isinstance(raw_output, tuple) and len(raw_output) == 2:
            # pyannote 3.x with return_embeddings=True path — kept for
            # compatibility if someone forces the model env back to 3.1.
            annotation, mean_embeddings = raw_output
            if exclusive_speaker_mode:
                logger.warning(
                    "exclusive_speaker_mode=True requested but pipeline "
                    "returned 3.x tuple (no exclusive_speaker_diarization). "
                    "Falling back to standard annotation."
                )
        else:  # pragma: no cover - 3.x without embeddings
            annotation = raw_output
            mean_embeddings = None
            if exclusive_speaker_mode:
                logger.warning(
                    "exclusive_speaker_mode=True requested but pipeline "
                    "returned bare Annotation (3.x legacy). Falling back."
                )

        # Observability: log which annotation branch ran. After the 2026-06-07
        # qualreai audit (workflow wf_5c42b72b), we want a definitive trace
        # in pod logs proving that exclusive_speaker_mode actually took the
        # exclusive_speaker_diarization branch in production — and that None
        # segments seen downstream come from silence gaps, not from this code
        # silently dropping the flag.
        source_label = (
            "exclusive"
            if (
                exclusive_speaker_mode
                and hasattr(raw_output, "exclusive_speaker_diarization")
                and annotation is getattr(
                    raw_output, "exclusive_speaker_diarization", None
                )
            )
            else "with_overlap"
        )
        try:
            track_count = sum(1 for _ in annotation.itertracks())
            label_set = sorted(annotation.labels())
        except Exception:  # noqa: BLE001
            track_count = -1
            label_set = []
        logger.info(
            "Diarization annotation source=%s, requested_exclusive=%s, "
            "tracks=%d, labels=%s",
            source_label,
            exclusive_speaker_mode,
            track_count,
            label_set,
        )

        # Collect turns, filtering by min_duration. Preserve start-time order.
        raw_turns: list[SpeakerTurn] = []
        label_durations: dict[str, float] = {}
        for segment, _, label in annotation.itertracks(yield_label=True):
            dur = float(segment.end) - float(segment.start)
            if dur < min_duration:
                continue
            label_str = str(label)
            raw_turns.append(
                SpeakerTurn(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker_label=label_str,
                )
            )
            label_durations[label_str] = label_durations.get(label_str, 0.0) + dur
        raw_turns.sort(key=lambda t: t.start)

        # pyannote returns embeddings in the order of annotation.labels(), which
        # is the cluster-id order — same as the SPEAKER_NN suffix order. In
        # pyannote 4.x the regular and exclusive annotations share labels
        # (they're renamed via the same mapping inside .apply()), so this
        # index lookup works regardless of which annotation we picked above.
        pipeline_labels = list(annotation.labels())
        speaker_stats: dict[str, SpeakerStats] = {}
        if mean_embeddings is not None:
            for i, label in enumerate(pipeline_labels):
                if label not in label_durations:
                    # Speaker had only short turns that we filtered out.
                    continue
                try:
                    emb_row = mean_embeddings[i]
                except (IndexError, TypeError):
                    logger.warning(
                        "Embedding index %d missing for label %s", i, label
                    )
                    continue
                arr = np.asarray(emb_row, dtype=np.float32).flatten()
                if arr.size == 0 or np.isnan(arr).any():
                    logger.warning(
                        "Skipping speaker %s with empty/NaN embedding", label
                    )
                    continue
                speaker_stats[str(label)] = SpeakerStats(
                    label=str(label),
                    seconds_observed=float(label_durations[label]),
                    mean_embedding=arr.tolist(),
                )

        seg_embs: Optional[list[dict]] = None
        if return_segment_embeddings:
            seg_embs = []
            emb_inf = _get_embedding_inference(pipeline)
            for turn in raw_turns:
                try:
                    # Same WAV path used for the diarization run — keeps
                    # the per-turn embeddings in the same sample-aligned
                    # space and avoids re-tripping the MP3 bug.
                    e = emb_inf.crop(
                        wav_path, Segment(turn.start, turn.end)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Per-turn embedding crop failed for %.2f-%.2f (%s): %s",
                        turn.start,
                        turn.end,
                        turn.speaker_label,
                        exc,
                    )
                    continue
                arr = np.asarray(e, dtype=np.float32).flatten()
                if arr.size == 0 or np.isnan(arr).any():
                    continue
                seg_embs.append(
                    {
                        "start": turn.start,
                        "end": turn.end,
                        "speaker_label": turn.speaker_label,
                        "vector": arr.tolist(),
                    }
                )

    return DiarizationResult(
        turns=raw_turns,
        num_speakers_detected=len(speaker_stats),
        diarization_model=DIARIZATION_MODEL,
        speaker_embeddings=speaker_stats,
        segment_embeddings=seg_embs,
    )
