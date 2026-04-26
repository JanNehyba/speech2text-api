"""Whisper transcription using Hugging Face's standard ASR pipeline.

QualReAI fork notes
-------------------
The original upstream defined a custom ``WhisperPipeline`` subclass to force
the output language and to limit ``max_length`` — both worked around quirks
in transformers 4.23 (October 2022). Modern transformers (>=4.45) support
language forcing via ``generate_kwargs={"language": ..., "task": "transcribe"}``
and chunked timestamps via ``return_timestamps=True``, so we no longer need
the subclass and use the off-the-shelf pipeline. This both simplifies the
code and unlocks segment-level timestamps that the older path could not
produce.
"""

import logging
from typing import Any, Dict, List, Optional

import librosa
from transformers import pipeline

from speech2text_api.speech2text_abs import Speech2Text

logger = logging.getLogger()

# Whisper's feature extractor always expects 16 kHz mono — resample upfront so
# the HF pipeline doesn't have to (and so a mismatched sample rate doesn't
# silently corrupt the input features).
WHISPER_SAMPLE_RATE = 16000


class Whisper(Speech2Text):

    def __init__(self, model_name_or_path: str = 'openai/whisper-tiny'):
        if model_name_or_path == 'openai/whisper-tiny':
            logger.warning(
                "You are using the default smallest whisper model, intended only for CI tests. "
                "Consider changing the default `model_name_or_path`, e.g., to `openai/whisper-large-v3-turbo`."
            )
        # chunk_length_s + stride_length_s let HF chunk long audio internally
        # with overlap so we don't lose context at chunk boundaries — much
        # better than the upstream's naïve 30 s slicing without overlap.
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name_or_path,
            chunk_length_s=30,
            stride_length_s=5,
        )

    def transcribe_file(
        self,
        fpath: str,
        lang: str = "cs",
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """Transcribe an audio file.

        Returns a dict with:
          - ``transcript``: full text (always present)
          - ``segments``: list of {start, end, text} dicts when
            ``return_timestamps=True`` and the model emits usable timestamps,
            otherwise ``None``.
        """
        generate_kwargs = {"language": lang, "task": "transcribe"}

        # Pre-decode with librosa instead of letting the HF pipeline run
        # ``ffmpeg_read`` over the raw bytes. ffmpeg_read is strict about
        # container format detection and raises ValueError on perfectly
        # valid m4a/mp3 files coming out of common phone recorders;
        # librosa+audioread+soundfile handle the same files. Pass the raw
        # numpy array straight to the pipeline so ffmpeg_read is bypassed.
        np_seq, sr = librosa.load(fpath, sr=WHISPER_SAMPLE_RATE, mono=True)
        inputs = {"raw": np_seq, "sampling_rate": sr}

        result = self.pipe(
            inputs,
            return_timestamps=return_timestamps if return_timestamps else False,
            generate_kwargs=generate_kwargs,
        )

        transcript: str = result["text"] if isinstance(result, dict) else str(result)

        segments: Optional[List[Dict[str, Any]]] = None
        if return_timestamps and isinstance(result, dict) and "chunks" in result:
            segments = []
            for chunk in result["chunks"]:
                ts = chunk.get("timestamp") if isinstance(chunk, dict) else None
                text = chunk.get("text") if isinstance(chunk, dict) else None
                if not text or not isinstance(text, str) or not text.strip():
                    continue
                if not isinstance(ts, (list, tuple)) or len(ts) != 2:
                    continue
                start, end = ts[0], ts[1]
                # The trailing chunk of long files often has a None end —
                # those segments aren't usable for audio navigation, drop them.
                if start is None or end is None:
                    continue
                segments.append({
                    "start": float(start),
                    "end": float(end),
                    "text": text,
                })

        return {"transcript": transcript, "segments": segments}
