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
# pod restart. The Dockerfile snapshot-downloads pyannote weights below; runtime
# pod runs as non-root (UID 1000 in ped-aivr-ns), so the cache dir must be
# world-writable (mode 1777, sticky bit) — otherwise the first HF call from the
# non-root user crashes with EACCES.
ENV HF_HOME=/opt/hf-cache
ENV TRANSFORMERS_CACHE=/opt/hf-cache
ENV NUMBA_CACHE_DIR=/tmp
RUN mkdir -p /opt/hf-cache && chmod 1777 /opt/hf-cache

# Pre-fetch pyannote.audio model weights at build time so the image is
# self-contained and the first /diarize call doesn't pay the download
# latency. Both pyannote 3.1 and 4.x community-1 weights are baked in so
# the pod can switch via PYANNOTE_DIARIZATION_MODEL without rebuild.
#
# community-1 also requires EULA acceptance on huggingface.co under the
# same account as the existing 3.1 / segmentation-3.0 / wespeaker EULAs.
# If the EULA isn't accepted yet, prefetch logs a warning and skips that
# model; the build still succeeds, and the runtime will retry the
# download lazily (provided HF_TOKEN env is set on the pod).
#
# The HF_TOKEN secret is mounted via BuildKit so it never appears in
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
COPY docker/prefetch_pyannote.py /tmp/prefetch_pyannote.py
ARG SKIP_PYANNOTE_PREFETCH=0
RUN --mount=type=secret,id=hf_token,required=false \
    if [ "$SKIP_PYANNOTE_PREFETCH" != "1" ] && [ -s /run/secrets/hf_token ]; then \
        HF_TOKEN="$(cat /run/secrets/hf_token)" python /tmp/prefetch_pyannote.py; \
    else \
        echo 'Skipping pyannote prefetch (no hf_token secret); will download lazily at runtime if HF_TOKEN env is set.'; \
    fi && \
    rm -f /tmp/prefetch_pyannote.py && \
    chmod -R 1777 /opt/hf-cache
# Recursive chmod above: snapshot_download creates subdirs as root (the build
# user). The runtime container runs as UID 1000 in ped-aivr-ns; without
# recursive 1777 the runtime user gets EACCES when faster-whisper (or any
# subsequent HF download) tries to mkdir under /opt/hf-cache/hub.

EXPOSE 5000
CMD ["speech2text-api", "httpapi", "--host", "0.0.0.0", "--port", "5000"]
