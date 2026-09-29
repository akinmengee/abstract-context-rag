# core/

Infrastructure with no RAG logic in it: settings, error types, logging,
database wiring, and auth primitives - the plumbing every other package
depends on.

| File | Holds | Used by |
|---|---|---|
| `config.py` | `Settings` — one typed class per config section (`llm`, `embedding`, `reranker`, `qdrant`, `chunking`, `retrieval`, `ingestion`, `database`, `auth`), loaded from `backend/config.yaml` and overridable with `ACR_*` env vars | `container.py`, `db.py`, `security.py`, and anything that reads a setting directly |
| `container.py` | `build_engine()` / `get_engine()` — the one place that constructs a `RagEngine` from `Settings` | `api/dependencies.py`, `cli.py` |
| `db.py` | `get_db_engine()` / `init_db()` — the SQLite engine for accounts and chat history, separate from `database/qdrant_store.py` (chunks/embeddings) | `api/dependencies.py`, `main.py`'s lifespan |
| `security.py` | Password hashing (`bcrypt`) and JWT issuance/verification (`pyjwt`) | `api/auth.py`, `api/dependencies.py::get_current_user` |
| `errors.py` | `RagError` and its subclasses: `SourceNotFoundError`, `FetchError`, `ParseError`, `InvalidInputError`, `LLMError` (engine-side), plus `InvalidCredentialsError`, `TokenExpiredError`, `EmailAlreadyRegisteredError`, `ConversationNotFoundError` (accounts-side) | raised anywhere under `rag/` or `api/`, caught once in `main.py` |
| `logging.py` | `setup_logging()` / `get_logger()` — one log format for the whole process | `cli.py`, `main.py` |

One exception to "core/ depends on nothing else in this project": `db.py`'s
`init_db()` imports `abstractrag.accounts.models`, to register its tables on
`SQLModel.metadata` before `create_all()` runs. The only place this folder
reaches outward, and only for that.

## Why this exists

- **One source of truth for configuration.** A setting is read in exactly one
  place (`config.py`); nothing else parses YAML or reads an env var directly.
  `ACR_LLM__MODEL=qwen3-8b` overrides `llm.model` from the shell without editing
  a file — useful for CI and Docker.
- **One source of truth for construction.** `container.py` is the only code that
  knows every component the engine is built from. Swapping an implementation
  (a different reranker, say) means changing one function, not every call site.
  `get_engine()` is cached, so the API and a long-running CLI command reuse the
  same instance instead of reloading models per call.
- **Errors carry meaning, not just a message.** `rag/` never raises a bare
  `Exception` or an HTTP status code — it raises the specific `RagError`
  subclass for what went wrong. `main.py` maps each subclass to a status code in
  one place, so a route handler never needs its own try/except.
- **Logging is configured once.** Any module calls `get_logger(__name__)`
  without caring whether it is running inside the API, the CLI, or a script.
- **Auth and DB access each have exactly one entry point.** Every password
  check and every token goes through `security.py`; every SQLite connection
  goes through `db.py::get_db_engine()`. Nothing under `api/` opens its own.
