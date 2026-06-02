"""Speaker diarization via pyannote.audio 3.1.

CPU-only inference. Models are loaded lazily on the first request so the
container can pass liveness probes during start-up without paying the
~12 MB segmentation + ~12 MB wespeaker embedding download cost up-front
(the Dockerfile snapshot-downloads them at build time, so first-request
loading is just a torch ``state_dict`` load from disk, sub-second).

Concurrency: pyannote peaks at ~2-3 GiB of working memory and saturates
all CPU cores. The MU pod runs faster-whisper alongside; running two
diarizations in parallel would OOM the pod (12 GiB limit) and starve the
Whisper path. ``get_concurrency_semaphore()`` returns a single-permit
asyncio semaphore that the HTTP layer must hold around the synchronous
diarize call.

For the cross-transcript voice-ID feature in QualReAI we return two kinds
of embeddings:

* ``speaker_embeddings`` — one mean vector per detected speaker, weighted
  by total speech duration. Used as the long-lived "voice profile".
* ``segment_embeddings`` (opt-in via ``return_segment_embeddings=True``)
  — one vector per turn, persisted on ``Segment`` rows so merge/split
  operations can recompute profiles exactly without re-running pyannote.
  All vectors come from the *same* embedding model the diarization
  pipeline uses internally, so cosine similarities are directly
  comparable across both shapes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from pyannote.audio import Inference, Model, Pipeline
from pyannote.core import Segment

logger = logging.getLogger(__name__)

DIARIZATION_MODEL = os.environ.get(
    "PYANNOTE_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
)
# Fallback embedding model used only if ``pipeline.embedding`` isn't exposed
# by the pyannote version we end up with. Matches the model 3.1 uses
# internally so per-turn embeddings stay comparable to the pipeline output.
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
            pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                use_auth_token=hf_token,
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
    occupy the same space as the mean-per-speaker vectors returned by
    ``pipeline(..., return_embeddings=True)``. Falls back to a canonical
    pyannote model that 3.1 ships with.
    """
    global _embedding_inference
    if _embedding_inference is not None:
        return _embedding_inference
    with _init_lock:
        if _embedding_inference is None:
            # pyannote 3.x exposes the underlying embedding model on the
            # pipeline. Attribute name has bounced between versions.
            emb_model = (
                getattr(pipeline, "embedding", None)
                or getattr(pipeline, "_embedding", None)
            )
            if emb_model is None:
                hf_token = os.environ.get("HF_TOKEN") or None
                logger.info(
                    "Pipeline did not expose embedding model; loading "
                    "fallback %s",
                    FALLBACK_EMBEDDING_MODEL,
                )
                emb_model = Model.from_pretrained(
                    FALLBACK_EMBEDDING_MODEL,
                    use_auth_token=hf_token,
                )
                emb_model.to(torch.device("cpu"))
            if isinstance(emb_model, Inference):
                _embedding_inference = emb_model
            else:
                _embedding_inference = Inference(emb_model, window="whole")
    return _embedding_inference


def diarize_audio_file(
    audio_path: str,
    num_speakers: int = 2,
    min_duration: float = 0.5,
    return_segment_embeddings: bool = False,
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
    """
    pipeline = _load_pipeline()

    annotation_result = pipeline(
        audio_path,
        num_speakers=num_speakers,
        return_embeddings=True,
    )

    # pyannote 3.x: pipeline(..., return_embeddings=True) returns a tuple
    # (Annotation, np.ndarray of shape (num_clusters, embed_dim)).
    if (
        isinstance(annotation_result, tuple)
        and len(annotation_result) == 2
    ):
        annotation, mean_embeddings = annotation_result
    else:  # pragma: no cover - shouldn't happen on 3.1
        annotation = annotation_result
        mean_embeddings = None

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
    # is the cluster-id order — same as the SPEAKER_NN suffix order in
    # speaker-diarization-3.1. We line them up by index.
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
                e = emb_inf.crop(
                    audio_path, Segment(turn.start, turn.end)
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
