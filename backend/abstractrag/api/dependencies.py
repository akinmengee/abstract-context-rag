"""FastAPI dependencies. The API only ever reaches the engine or the
database through this."""

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import Engine
from sqlmodel import Session

from abstractrag.accounts.models import User
from abstractrag.core.config import get_settings
from abstractrag.core.container import get_engine
from abstractrag.core.db import get_db_engine
from abstractrag.core.errors import InvalidCredentialsError
from abstractrag.core.security import decode_access_token
from abstractrag.rag.engine import RagEngine

EngineDep = Annotated[RagEngine, Depends(get_engine)]

# Routed through Depends() (not called directly) so tests can override it the
# same way SessionDep is overridden - used by api/chat.py's streaming
# persistence, which needs its own Engine rather than the request-scoped
# SessionDep (see that module for why).
DbEngineDep = Annotated[Engine, Depends(get_db_engine)]

_bearer_scheme = HTTPBearer(auto_error=False)


def get_session() -> Generator[Session, None, None]:
    with Session(get_db_engine()) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def get_current_user(
    session: SessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> User:
    if credentials is None:
        raise InvalidCredentialsError("missing bearer token")
    user_id = decode_access_token(credentials.credentials, get_settings().auth)
    user = session.get(User, user_id)
    if user is None:
        raise InvalidCredentialsError("user not found")
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]
