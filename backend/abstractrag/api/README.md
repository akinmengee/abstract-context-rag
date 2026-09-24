# api/

The HTTP layer. Every file here is a thin shell: parse the request, call the
engine, return the result. No RAG logic lives in this folder — that all lives in
`rag/engine.py`, which this layer only calls.

| File | Route(s) | Does |
|---|---|---|
| `dependencies.py` | — | `EngineDep`: the one way a route gets a `RagEngine`. Backed by `core/container.py`, cached so every request shares the same engine instance. |
| `schemas.py` | — | Pydantic **request** models (`IngestRequest`, `ChatRequest`) plus `DocumentSummary`. Named after the HTTP request shape, not the database — Qdrant's own schema lives in `database/qdrant_store.py` and this file has no relation to it. Kept here, not as its own top-level package, because nothing outside `api/` uses it (confirm with a grep before moving it back out). |
| `router.py` | — | Mounts every route below under `/api/v1` and wires them into one `APIRouter`. `main.py` includes only this. |
| `health.py` | `GET /health`, `GET /health/dependencies` | Process liveness, and whether Qdrant/the LLM are reachable. |
| `ingest.py` | `POST /ingest`, `POST /ingest/upload` | Ingest by arXiv ID or Wikipedia title/URL (JSON body), or by uploading a PDF (multipart). Both end in the same `engine.ingest()` call. |
| `documents.py` | `GET /documents` | Lists what has been ingested. |
| `chat.py` | `POST /chat`, `POST /chat/stream`, `POST /summarize` | Ask a question; the second endpoint streams the answer as Server-Sent Events. `/chat` automatically routes global questions ("summarize this paper") to the same map-reduce summarisation `/summarize` calls directly. |

There is no `v1/` subfolder on purpose: with a single API version, the extra
directory added no value. `/api/v1` still comes from the prefix set in
`router.py`, so a future `v2` only needs new route files plus a second prefixed
router — nothing here has to move.

## Adding a route

1. Create (or reuse) a module here with an `APIRouter()`.
2. Add the request shape to `schemas.py` (responses reuse `rag/models.py` directly).
3. Call exactly one `engine.*` method — if the handler needs more than that, the
   logic belongs in `rag/engine.py`, not here.
4. Register the router in `router.py`.

Errors raised anywhere under `rag/` (subclasses of `core.errors.RagError`) are
caught once, centrally, in `main.py` and mapped to HTTP status codes — route
handlers never need their own try/except.
