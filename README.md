## Speech2Text API — QualReAI fork

Fork of [EduMUNI/speech2text-api](https://github.com/EduMUNI/speech2text-api)
that adds optional segment-level timestamps to the transcription endpoint
for use as the Whisper sidecar in
[QualReAI](https://github.com/JanNehyba/QualReAI).

### Differences from upstream

- `?timestamps=true` query parameter on `/transcribe/{lang}/` returns
  `{transcript, segments: [{start, end, text}, ...]}` — segments come from
  Hugging Face's chunked ASR pipeline (`return_timestamps=True`,
  `chunk_length_s=30`, `stride_length_s=5`). Default behavior (no query
  param) is wire-compatible with the original upstream.
- Custom `WhisperPipeline` subclass dropped; standard HF
  `pipeline("automatic-speech-recognition", ...)` is used directly because
  newer transformers (>=4.45) handles language forcing and timestamps
  natively.
- Python 3.11, transformers >=4.45, torch >=2.2.
- `sentry-asgi` (unmaintained) replaced by `sentry-sdk`'s built-in ASGI
  middleware, applied only when `SPEECH2TEXT_API_SENTRY_DSN` is set.
- GitHub Actions workflow publishes the image to
  `ghcr.io/jannehyba/speech2text-api:latest` on push to master.

### Usage

```shell
# Plain transcript (legacy):
curl -F "file=@tests/res/test_cs_longer.wav" \
     http://localhost:5000/transcribe/cs/

# With segment timestamps:
curl -F "file=@tests/res/test_cs_longer.wav" \
     "http://localhost:5000/transcribe/cs/?timestamps=true"
```

Response with `?timestamps=true`:

```json
{
  "transcript": "...",
  "segments": [
    {"start": 0.0, "end": 4.2, "text": " ..."},
    {"start": 4.2, "end": 8.7, "text": " ..."}
  ]
}
```

### As a Python library

```python
from speech2text_api.whisper import Whisper

wrapper = Whisper(model_name_or_path="openai/whisper-large-v3-turbo")
result = wrapper.transcribe_file(
    "tests/res/test_cs_longer.wav",
    lang="cs",
    return_timestamps=True,
)
print(result["transcript"])
for seg in result["segments"]:
    print(f"{seg['start']:.1f}-{seg['end']:.1f}: {seg['text']}")
```
