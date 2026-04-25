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

# Install Python deps first so changes to source code reuse the layer cache.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy source and install the package in non-editable mode.
COPY setup.py README.md /app/
COPY src /app/src
# setup.py uses setuptools_scm which expects a .git/ — fall back to a pinned
# version when building outside a git checkout (e.g. inside Docker layer).
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
RUN pip install --no-cache-dir /app

# Cache locations for HF model and Numba JIT — must be writable at runtime.
ENV NUMBA_CACHE_DIR=/tmp/
ENV TRANSFORMERS_CACHE=/tmp/
ENV HF_HOME=/tmp/huggingface

EXPOSE 5000
CMD ["speech2text-api", "httpapi", "--host", "0.0.0.0", "--port", "5000"]
