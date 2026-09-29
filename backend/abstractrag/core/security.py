"""Password hashing and JWT issuance/verification.

bcrypt directly, not passlib: passlib is unmaintained and breaks against
recent bcrypt versions - bcrypt alone does the one thing needed here.
pyjwt, not python-jose: smaller, and one HS256 token needs no extra surface.
"""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from abstractrag.core.config import AuthSettings
from abstractrag.core.errors import InvalidCredentialsError, TokenExpiredError


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, settings: AuthSettings) -> str:
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": user_id, "exp": expires_at}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str, settings: AuthSettings) -> str:
    """Return the user id (`sub`) a valid token carries."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("access token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise InvalidCredentialsError("invalid access token") from exc
    return payload["sub"]
