# core/

Infrastructure with no RAG logic in it: settings, error types, logging, and the
wiring that turns settings into a running engine. Every other package depends on
`core/`; `core/` depends on nothing else in this project.

| File | Holds | Used by |
|---|---|---|
| `config.py` | `Settings` — one typed class per config section (`llm`, `embedding`, `reranker`, `qdrant`, `chunking`, `retrieval`, `ingestion`), loaded from `backend/config.yaml` and overridable with `ACR_*` env vars | `container.py`, and anything that reads a setting directly |
| `container.py` | `build_engine()` / `get_engine()` — the one place that constructs a `RagEngine` from `Settings` | `api/dependencies.py`, `cli.py` |
| `errors.py` | `RagError` and its subclasses (`SourceNotFoundError`, `FetchError`, `ParseError`, `InvalidInputError`, `LLMError`) | raised anywhere under `rag/`, caught once in `main.py` |
| `logging.py` | `setup_logging()` / `get_logger()` — one log format for the whole process | `cli.py`, `main.py`, `scripts/` |

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
