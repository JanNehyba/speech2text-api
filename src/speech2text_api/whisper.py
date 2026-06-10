"""Whisper transcription using faster-whisper (CTranslate2 backend).

QualReAI fork — speed-optimized variant
---------------------------------------
The first iteration of this fork used Hugging Face's ``transformers`` ASR
pipeline. On CPU that turned out to be ~2× slower than the original upstream
(transformers 4.45 has a much heavier per-chunk hot path than 4.23 had, and
HF's stride-based chunking adds further overhead).

This rewrite replaces the HF pipeline with `faster-whisper`, a CTranslate2
reimplementation of OpenAI's Whisper inference. On CPU it runs ~4–8× faster
than ``transformers`` at the same model size, with no measurable accuracy
loss for transcription tasks. Combined with int8 quantization and greedy
decoding (``beam_size=1``) we end up roughly an order of magnitude faster
than the original upstream, while gaining native segment-level timestamps
for free.

The public ``transcribe_file`` contract stays the same:
``{transcript, segments | None}``, so ``httpapi.py`` and the QualReAI client
don't have to change.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import librosa
from faster_whisper import WhisperModel

from speech2text_api.speech2text_abs import Speech2Text

logger = logging.getLogger()

# Whisper feature extractor expects 16 kHz mono. Resample upfront with librosa
# (matches the legacy upstream behaviour and survives weird m4a/mp3 headers
# that confuse ffmpeg_read).
WHISPER_SAMPLE_RATE = 16000

# faster-whisper accepts both its own short aliases and HF IDs. We map the
# common HF whisper IDs to aliases so the pre-converted CT2 model gets pulled
# instead of triggering a slow on-the-fly conversion.
_HF_TO_FW_ALIAS = {
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v2": "large-v2",
    "openai/whisper-medium": "medium",
    "openai/whisper-small": "small",
    "openai/whisper-base": "base",
    "openai/whisper-tiny": "tiny",
}


def _resolve_model(model_name_or_path: str) -> str:
    return _HF_TO_FW_ALIAS.get(model_name_or_path, model_name_or_path)


class Whisper(Speech2Text):

    def __init__(self, model_name_or_path: str = "openai/whisper-tiny"):
        resolved = _resolve_model(model_name_or_path)
        if resolved == "tiny":
            logger.warning(
                "Using the tiny Whisper model — intended for CI tests only. "
                "Set SPEECH2TEXT_API_MODEL_ID=openai/whisper-large-v3-turbo "
                "(or large-v3-turbo) for production-grade transcription."
            )

        # Speed knobs — exposed via env so deployments can tune without a
        # rebuild. Defaults are picked for max speed at acceptable quality:
        #   compute_type=int8        ~2-3× faster than float32 on CPU,
        #                            <1% WER hit for transcription
        #   cpu_threads=OMP_NUM_THREADS or 4
        #   num_workers=1            (single audio at a time is the common case)
        compute_type = os.environ.get("FW_COMPUTE_TYPE", "int8")
        cpu_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))

        logger.info(
            "Loading faster-whisper model=%s compute_type=%s cpu_threads=%d",
            resolved, compute_type, cpu_threads,
        )
        self.model = WhisperModel(
            resolved,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=1,
        )

        # beam_size=1 is greedy decoding — ~3-5× faster than the default 5,
        # with a small accuracy hit. Override per-deployment if quality drops.
        self.beam_size = int(os.environ.get("FW_BEAM_SIZE", "1"))
        # VAD filter skips silent regions before inference. For interview audio
        # with pauses this is a free speedup; can be disabled if it cuts off
        # quiet speech.
        self.vad_filter = os.environ.get("FW_VAD_FILTER", "true").lower() in (
            "1", "true", "yes", "on",
        )

    def transcribe_file(
        self,
        fpath: str,
        lang: str = "cs",
        return_timestamps: bool = False,
        beam_size: Optional[int] = None,
        vad_filter: Optional[bool] = None,
        word_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """Transcribe an audio file.

        ``beam_size`` and ``vad_filter`` are per-call overrides for the
        env-configured defaults — clients (e.g. QualReAI's quality preset
        selector) can pick speed vs. quality without restarting the server.
        Pass ``None`` to use the env defaults.

        ``word_timestamps`` opts into per-word timestamps. faster-whisper
        computes these via a DTW over the model's cross-attention weights —
        +10-30% inference time on CPU int8, so it stays OFF by default and
        only the diarization-aware client requests it (QualReAI uses the
        per-word grid to split a Whisper segment at a speaker boundary so a
        short interviewer question doesn't get swallowed into the
        respondent's answer).

        Returns:
          - ``transcript``: full text (always present)
          - ``segments``: list of {start, end, text} when ``return_timestamps``
            is True, otherwise ``None``. faster-whisper always emits segments
            with timestamps internally — we just decide whether to expose them.
          - ``words``: flat list of {word, start, end} when ``word_timestamps``
            is True, otherwise ``None``.
        """
        effective_beam = beam_size if beam_size is not None else self.beam_size
        effective_vad = vad_filter if vad_filter is not None else self.vad_filter

        # Decode to 16 kHz mono with librosa — same as before to avoid
        # ffmpeg_read pickiness on phone-recorded m4a/mp3 files.
        np_seq, _sr = librosa.load(fpath, sr=WHISPER_SAMPLE_RATE, mono=True)

        segments_iter, _info = self.model.transcribe(
            np_seq,
            language=lang,
            beam_size=effective_beam,
            vad_filter=effective_vad,
            word_timestamps=word_timestamps,
        )

        # The iterator only runs inference as it's consumed. Materialize once
        # so we can both join the text and (optionally) emit segments.
        segments_list: List[Any] = list(segments_iter)

        transcript = "".join(s.text for s in segments_list).strip()

        segments_payload: Optional[List[Dict[str, Any]]] = None
        if return_timestamps:
            segments_payload = [
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text,
                }
                for s in segments_list
                if s.start is not None and s.end is not None and s.text
            ]

        words_payload: Optional[List[Dict[str, Any]]] = None
        if word_timestamps:
            words_payload = []
            for s in segments_list:
                for w in (getattr(s, "words", None) or []):
                    if w.start is None or w.end is None or not w.word:
                        continue
                    words_payload.append(
                        {
                            "word": w.word,
                            "start": float(w.start),
                            "end": float(w.end),
                        }
                    )

        return {
            "transcript": transcript,
            "segments": segments_payload,
            "words": words_payload,
        }
