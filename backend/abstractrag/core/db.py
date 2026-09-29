"""SQLite persistence for accounts and chat history, via SQLModel.

Separate from database/qdrant_store.py (the vector store for chunks and
embeddings) - this is relational storage for who's asking and what they've
asked before.

No Alembic: three tables (User, Conversation, Message), no rows in
production yet. SQLModel.metadata.create_all() is enough until there's an
actual migration to write. Deliberate simplicity, not an oversight.
"""

from functools import lru_cache

from sqlalchemy import Engine
from sqlmodel import SQLModel, create_engine

from abstractrag.core.config import get_settings


@lru_cache
def get_db_engine() -> Engine:
    settings = get_settings()
    path = settings.database.resolved_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI runs sync path operations in a
    # threadpool - each request's session (api/dependencies.py::get_session)
    # checks out its own connection from this one shared Engine, the
    # standard documented SQLModel+FastAPI pattern for exactly this case.
    return create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})


def init_db() -> None:
    """Create tables that don't exist yet. Called once from main.py's lifespan."""
    import abstractrag.accounts.models  # noqa: F401 - registers tables on SQLModel.metadata

    SQLModel.metadata.create_all(get_db_engine())
