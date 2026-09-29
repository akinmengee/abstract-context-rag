# accounts/

User accounts and chat history — SQLModel tables, persisted to the small
SQLite database `core/db.py` manages. Separate from `database/qdrant_store.py`
(document chunks/embeddings, a shared pool); this folder is per-user state
only - who's asking, and what they've asked before.

| File | Holds |
|---|---|
| `models.py` | `User`, `Conversation`, `Message` (+ `MessageRole`) - the three tables, plus the `_uuid()`/`_now()` default factories they share |

## Notes

- String UUID primary keys, not autoincrement ints - matches the rest of the
  project's convention (`rag/models.py`'s `chunk_id_for`/`document_id_for`).
- `Conversation.document_id: str | None` - `None` means no restriction, the
  existing multi-document/agentic mode (`RagEngine.ask(document_id=None)`).
  Fixed once at creation, never changed afterward.
- `Message.citations` is stored as JSON (`Answer.citations` after
  `.model_dump()`); `None` for a user message, or an assistant message that
  cited nothing.
- No Alembic: three tables, no migration history yet -
  `SQLModel.metadata.create_all()` (called from `core/db.py::init_db()`) is
  enough until there actually is one to write.

Used by `api/auth.py` (register/login), `api/conversations.py` (CRUD) and
`api/chat.py` (reading/writing messages) - tested via
`tests/unit/test_auth.py` and `tests/unit/test_conversations.py`.
