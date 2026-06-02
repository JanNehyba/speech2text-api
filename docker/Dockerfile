# Speech2Text API — QualReAI fork
# Built from source (no external release.tar.gz dependency) so CI can build
# directly from a git checkout.
FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash curl ffmpeg gcc git libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install non-torch Python deps first so changes to source code reuse the
# layer cache.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.* torchaudio==2.4.* && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy source and install the package in non-editable mode.
COPY setup.py README.md /app/
COPY src /app/src
# setup.py uses setuptools_scm which expects a .git/ — fall back to a pinned
# version when building outside a git checkout (e.g. inside Docker layer).
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
RUN pip install --no-cache-dir /app

# Persistent HF cache so pyannote / faster-whisper don't re-download on every
# pod restart. The Dockerfile snapshot-downloads pyannote weights below.
ENV HF_HOME=/opt/hf-cache
ENV TRANSFORMERS_CACHE=/opt/hf-cache
ENV NUMBA_CACHE_DIR=/tmp

# Pre-fetch pyannote.audio model weights at build time so the image is
# self-contained and the first /diarize call doesn't pay the ~30 MB download
# latency. The HF_TOKEN secret is mounted via BuildKit so it never appears in
# any layer.
#
# Build with:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     -f docker/Dockerfile .
#
# Or skip the prefetch entirely (the runtime will download lazily, but the pod
# then needs HF_TOKEN in its env and the first request stalls):
#   docker build --build-arg SKIP_PYANNOTE_PREFETCH=1 ...
ARG SKIP_PYANNOTE_PREFETCH=0
RUN --mount=type=secret,id=hf_token,required=false \
    if [ "$SKIP_PYANNOTE_PREFETCH" != "1" ] && [ -s /run/secrets/hf_token ]; then \
        HF_TOKEN="$(cat /run/secrets/hf_token)" python -c "\
import os; \
from huggingface_hub import snapshot_download; \
tok = os.environ.get('HF_TOKEN'); \
snapshot_download('pyannote/speaker-diarization-3.1', token=tok); \
snapshot_download('pyannote/segmentation-3.0', token=tok); \
snapshot_download('pyannote/wespeaker-voxceleb-resnet34-LM', token=tok); \
print('pyannote weights cached at', os.environ['HF_HOME']);\
"; \
    else \
        echo 'Skipping pyannote prefetch (no hf_token secret); will download lazily at runtime if HF_TOKEN env is set.'; \
    fi

EXPOSE 5000
CMD ["speech2text-api", "httpapi", "--host", "0.0.0.0", "--port", "5000"]
