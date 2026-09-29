# api/

The HTTP layer. Every file here is a thin shell: parse the request, call the
engine, return the result. No RAG logic lives in this folder — that all lives in
`rag/engine.py`, which this layer only calls.

| File | Route(s) | Does |
|---|---|---|
| `dependencies.py` | — | `EngineDep`, `SessionDep`, `CurrentUserDep`, `DbEngineDep` — the only way a route gets a `RagEngine`, a DB session, the authenticated `User`, or a raw DB `Engine` (`chat.py`'s streaming persistence needs its own — see that file for why `SessionDep` won't do). `EngineDep` is backed by `core/container.py`, cached so every request shares the same engine instance. |
| `schemas.py` | — | Pydantic **request** models for ingest/chat/summarize (`IngestRequest`, `ChatRequest`, `SummarizeRequest`) plus `DocumentSummary`. `auth.py` and `conversations.py` define their own request/response models locally instead — nothing outside those two files uses them, so there was nothing to centralize. Named after the HTTP request shape, not the database — Qdrant's own schema lives in `database/qdrant_store.py` and this file has no relation to it. Kept here, not as its own top-level package, because nothing outside `api/` uses it (confirm with a grep before moving it back out). |
| `router.py` | — | Mounts every route below under `/api/v1` and wires them into one `APIRouter`. `main.py` includes only this. |
| `health.py` | `GET /health`, `GET /health/dependencies` | Process liveness, and whether Qdrant/the LLM are reachable. |
| `auth.py` | `POST /auth/register`, `POST /auth/login` | Issues a bearer JWT (`core/security.py`). Every other authenticated route below requires it via `CurrentUserDep`. |
| `ingest.py` | `POST /ingest`, `POST /ingest/upload` | Ingest by arXiv ID or Wikipedia title/URL (JSON body), or by uploading a PDF (multipart). Both end in the same `engine.ingest()` call. |
| `documents.py` | `GET /documents` | Lists what has been ingested - the shared document pool, not scoped per user (unauthenticated on purpose). |
| `conversations.py` | `POST/GET /conversations`, `GET/PATCH/DELETE /conversations/{id}` | CRUD, scoped to the current user (`CurrentUserDep`). A conversation's `document_id` (which paper, or `None` for the cross-paper agent mode) is fixed at creation and never changed. |
| `chat.py` | `POST /chat`, `POST /chat/stream`, `POST /summarize` | Ask a question inside an existing conversation - `conversation_id` and `CurrentUserDep` are both required, and the conversation's own `document_id` decides what it can draw on. `/chat/stream` sends the answer as Server-Sent Events. Both automatically route global questions ("summarize this paper") to the same map-reduce summarisation `/summarize` calls directly. |

There is no `v1/` subfolder on purpose: with a single API version, the extra
directory added no value. `/api/v1` still comes from the prefix set in
`router.py`, so a future `v2` only needs new route files plus a second prefixed
router — nothing here has to move.

## Adding a route

1. Create (or reuse) a module here with an `APIRouter()`.
2. Add the request shape to `schemas.py`, unless the route is about accounts
   or conversations - those define their own models locally (see the table
   above for why).
3. Call exactly one `engine.*` method — if the handler needs more than that, the
   logic belongs in `rag/engine.py`, not here. Responses reuse `rag/models.py`
   directly where they can.
4. Register the router in `router.py`.

Errors raised anywhere under `rag/` (subclasses of `core.errors.RagError`) are
caught once, centrally, in `main.py` and mapped to HTTP status codes — route
handlers never need their own try/except.
