"""Pre-fetch pyannote.audio weights into the image at build time.

Runs inside the Docker build with HF_TOKEN set from a BuildKit secret. If
the token holder hasn't accepted a model's EULA yet, that download fails
with a GatedRepoError — but we don't fail the whole build, we just log a
warning. The runtime pod will then download the model lazily on the first
/diarize request, using its own HF_TOKEN env var.

This lets the image build succeed even when a freshly created HF token
hasn't been authorized for all 3 models yet — the operator can accept
the EULAs, then re-run the workflow to bake them in.
"""

import os
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

MODELS = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var not set; skipping prefetch.", file=sys.stderr)
        return 0
    failures = []
    for model in MODELS:
        try:
            snapshot_download(model, token=token)
            print(f"OK: cached {model}")
        except GatedRepoError:
            failures.append(model)
            print(
                f"GATED (EULA not accepted yet): {model}", file=sys.stderr
            )
        except Exception as e:  # noqa: BLE001
            failures.append(model)
            print(
                f"FAILED: {model} {type(e).__name__}: {e}", file=sys.stderr
            )
    cache_dir = os.environ.get("HF_HOME", "(default)")
    if failures:
        print(
            f"WARN: {len(failures)} model(s) missing from image; "
            f"they'll download lazily at runtime if HF_TOKEN is set on the pod.",
            file=sys.stderr,
        )
    print(f"pyannote weights cached at {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
