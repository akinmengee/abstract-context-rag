FROM python:3.11-slim

# uv resolves and installs far faster than pip, which matters with torch-sized deps.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first so code edits do not invalidate the install layer.
# Hand-listed (not `uv pip install -e .`) so this layer doesn't need
# abstractrag/ copied in yet - keep this in sync with pyproject.toml's
# `dependencies` list.
COPY pyproject.toml ./
# torch (+ torchvision, its usual companion pin) first, from PyPI's CPU-only
# wheel index by default: FlagEmbedding pulls torch in transitively, and an
# unpinned install resolves the default CUDA build (cuBLAS/cuDNN/NCCL, ~1.5-2
# GB) even though this container normally runs with ACR_EMBEDDING__DEVICE=cpu
# / ACR_RERANKER__DEVICE=cpu (see below) - dead weight that only slows the
# image pull. torchvision has to be installed in the same command as torch -
# installing it separately later (e.g. as a transitive dependency pulled in
# by the next RUN) resolves it against PyPI's default (CUDA) index regardless
# of what torch is already present, producing a torchvision build with a
# mismatched ABI (`operator torchvision::nms does not exist` on import -
# caught by actually running the built image, not just by it building
# successfully).
#
# --build-arg TORCH_VARIANT=cuda (docker-compose.yml's `backend.build.args`
# sets this - it's the default there, see that file's header) skips the
# CPU-only index so both resolve PyPI's default CUDA build instead. This is
# not cosmetic: with a CPU-only torch, setting ACR_EMBEDDING__DEVICE=cuda
# doesn't silently fall back to CPU, it crashes the request with
# "AssertionError: Torch not compiled with CUDA enabled" the moment a model
# is moved to the device - reservation or not. Measured 2026-09-29: an
# earlier version of this Dockerfile always installed the CPU-only wheel
# with no ARG at all, while a since-removed second compose file added only
# the env vars and GPU reservation, never a torch rebuild - the one
# "verified" check that overlay ever got had run a host Python process (real
# CUDA torch, installed outside Docker), never the container itself, and
# missed this entirely. Both files are now one (docker-compose.yml, this
# ARG) instead of two.
ARG TORCH_VARIANT=cpu
RUN if [ "$TORCH_VARIANT" = "cuda" ]; then \
      uv pip install --system --no-cache-dir torch torchvision; \
    else \
      uv pip install --system --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi
RUN uv pip install --system --no-cache-dir fastapi uvicorn[standard] pydantic pydantic-settings \
    pyyaml python-multipart httpx typer qdrant-client docling FlagEmbedding arxiv numpy scikit-learn \
    sqlmodel bcrypt pyjwt email-validator

COPY abstractrag ./abstractrag
COPY config.yaml ./
RUN uv pip install --system --no-cache-dir --no-deps .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "abstractrag.main:app", "--host", "0.0.0.0", "--port", "8000"]
