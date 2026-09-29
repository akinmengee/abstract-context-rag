"""Registration and login. Issues a bearer JWT; every other authenticated
endpoint requires it via CurrentUserDep."""

from fastapi import APIRouter
from pydantic import BaseModel, EmailStr, Field
from sqlmodel import select

from abstractrag.accounts.models import User
from abstractrag.api.dependencies import SessionDep
from abstractrag.core.config import get_settings
from abstractrag.core.errors import EmailAlreadyRegisteredError, InvalidCredentialsError
from abstractrag.core.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)  # 72: bcrypt's own limit


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str


@router.post("/register", response_model=TokenResponse)
def register(request: RegisterRequest, session: SessionDep) -> TokenResponse:
    existing = session.exec(select(User).where(User.email == request.email)).first()
    if existing is not None:
        raise EmailAlreadyRegisteredError(f"{request.email} is already registered")
    user = User(email=request.email, hashed_password=hash_password(request.password))
    session.add(user)
    session.commit()
    session.refresh(user)
    return _issue_token(user)


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, session: SessionDep) -> TokenResponse:
    user = session.exec(select(User).where(User.email == request.email)).first()
    if user is None or not verify_password(request.password, user.hashed_password):
        raise InvalidCredentialsError("incorrect email or password")
    return _issue_token(user)


def _issue_token(user: User) -> TokenResponse:
    token = create_access_token(user.id, get_settings().auth)
    return TokenResponse(access_token=token, user_id=user.id, email=user.email)
