# database/

Storage access — code only. The data itself lives in `data/qdrant/` outside the
repo (gitignored), mounted as a Docker volume. Nothing else in the project talks
to Qdrant directly; every read and write goes through `QdrantStore`.

| File | Holds |
|---|---|
| `qdrant_store.py` | `QdrantStore` — the collection schema and every Qdrant operation the engine needs |

Note: `api/schemas.py` is unrelated to this folder despite the similar word —
it holds Pydantic **HTTP request** models, not a database schema. The two
"schema" concepts just happen to share a name.

## Schema

One collection, `settings.qdrant.collection` (default `documents`), holds two
named vectors per point:

- `dense` — the bge-m3 dense embedding, cosine distance
- `sparse` — the bge-m3 lexical weights, as a Qdrant sparse vector

Keeping them as two named vectors instead of one merged score is deliberate:
`search_dense()` and `search_sparse()` can be called and measured independently,
which is what the retrieval-mode ablation (dense vs sparse vs hybrid) needs.
Fusing them into one hybrid score happens one layer up, in
`rag/retrieval/hybrid.py`, via Reciprocal Rank Fusion — this file only returns
raw per-mode results.

The full `Chunk` (text + metadata) is stored as JSON in the point's payload, so a
search result can be turned back into a `Chunk` without a second lookup
(`_to_chunks`). A `document_id` payload index makes `delete_document()` and
per-document filtered search fast.

## API surface

`ensure_collection()`, `upsert_chunks()`, `delete_document()`, `search_dense()`,
`search_sparse()`, `list_documents()`. `RagEngine.ingest()` calls
`delete_document()` before `upsert_chunks()` on every ingest, so re-ingesting the
same source replaces it instead of duplicating it (chunk IDs are stable — see
`rag/models.py::chunk_id_for` — but re-chunking can change the count).

## Testing

Covered by `tests/integration/test_qdrant_store.py`, which runs against a real
Qdrant instance and skips itself when none is reachable — a storage layer is
exactly where a mock would hide real bugs (wrong filter, wrong vector name).
