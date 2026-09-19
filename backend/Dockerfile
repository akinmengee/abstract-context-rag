FROM python:3.11-slim

# uv resolves and installs far faster than pip, which matters with torch-sized deps.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first so code edits do not invalidate the install layer.
COPY pyproject.toml ./
RUN uv pip install --system --no-cache-dir fastapi uvicorn[standard] pydantic pydantic-settings \
    pyyaml python-multipart httpx typer qdrant-client docling FlagEmbedding arxiv

COPY abstractrag ./abstractrag
COPY config.yaml ./
COPY scripts ./scripts
RUN uv pip install --system --no-cache-dir --no-deps .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "abstractrag.main:app", "--host", "0.0.0.0", "--port", "8000"]
