# abstract-context-rag

A local, modular Retrieval-Augmented Generation (RAG) system for asking questions
about and summarizing research papers and Wikipedia articles. Every answer is
grounded in the source text, cited down to the page or section, and checked by a
verification step. If the source doesn't contain the answer, the system says so
instead of guessing.

## Goals

- **Grounded answers:** minimize hallucination, and measure it rather than claim it.
- **Learn by building:** implement each RAG technique from scratch, one at a time,
  and evaluate what it actually adds.
- **Fully local:** runs on consumer hardware (6 GB VRAM) with llama.cpp,
  quantized GGUF models, Qdrant, and local embedding/reranking models.
- **Modular engine:** the RAG core is independent of any interface, so the same
  engine can power a chatbot, a summarizer, or a literature review tool.

## What's inside

Implemented today:

- Section-aware PDF parsing (Docling layout analysis), arXiv fetching by ID, and
  Wikipedia ingestion — all through source adapters behind one `ingest()` entry point
- Hybrid search (bge-m3 dense + sparse) fused with Reciprocal Rank Fusion
- Cross-encoder reranking, plus a score threshold that abstains before the LLM is asked
- Grounded generation with numbered citations and lost-in-the-middle context ordering
- FastAPI backend with SSE streaming, an `abstractrag` CLI, and a Gradio dev console

Planned (see [docs/roadmap.md](docs/roadmap.md)):

- Query transformation (rewriting, HyDE, routing)
- Map-reduce and hierarchical (RAPTOR-style) summarization
- Citation verification with claim decomposition
- Agentic / corrective RAG for multi-document questions
- Evaluation harness with a golden dataset and ablation tables
- React and Flutter clients

## Quick start

Requires Docker, Python 3.11+, [uv](https://docs.astral.sh/uv/), and a
[llama.cpp](https://github.com/ggml-org/llama.cpp) server running a Qwen3 GGUF model.

```bash
# 1. Vector database
docker compose up -d qdrant

# 2. LLM on the host (keeps the GPU out of Docker)
llama-server -m models/qwen3-4b-q4_k_m.gguf --port 8080 --ctx-size 4096 -ngl 99

# 3. Backend
cd backend
uv pip install -e ".[dev]"
abstractrag serve                       # http://localhost:8000/docs

# 4. Ingest and ask
abstractrag ingest --arxiv 2005.11401
abstractrag ask "What retrieval setup does the paper use?"
```

The dev console (`cd ui && python app.py`) shows the retrieved chunks and their
scores next to each answer.

## Layout

```
backend/    FastAPI app + the RAG engine (abstractrag package)
ui/         Gradio dev console
web/        React web client (skeleton)
mobile/     Flutter client (skeleton)
shared/     Generated OpenAPI contract
docs/       Roadmap, architecture notes, ablation results
```

The engine is a plain Python library: `backend/abstractrag/rag/` has no
idea an API exists, so the CLI, the tests and the evaluation scripts drive the same
code the HTTP layer does.

## Configuration

`backend/config.yaml` holds the model, retrieval and chunking settings.
Any value can be overridden with an environment variable:

```bash
ACR_LLM__MODEL=qwen3-8b ACR_RETRIEVAL__MODE=dense abstractrag ask "..."
```

## Status

Work in progress. See [docs/roadmap.md](docs/roadmap.md) for phases and results.

## License

MIT
