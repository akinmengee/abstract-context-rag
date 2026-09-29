"""Conversation CRUD, scoped to the current user. Documents stay a shared/
global pool; only which conversation references which document, and its
messages, are scoped per user."""

from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import Session, select

from abstractrag.accounts.models import Conversation, Message
from abstractrag.api.dependencies import CurrentUserDep, SessionDep
from abstractrag.core.errors import ConversationNotFoundError

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    document_id: str | None = None  # None = "all documents"
    title: str | None = None  # defaults to "New Chat"


class ConversationUpdate(BaseModel):
    title: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    document_id: str | None
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict] | None
    created_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut]


@router.post("", response_model=ConversationSummary)
def create_conversation(
    request: ConversationCreate, user: CurrentUserDep, session: SessionDep
) -> Conversation:
    conversation = Conversation(
        user_id=user.id, document_id=request.document_id, title=request.title or "New Chat"
    )
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    return conversation


@router.get("", response_model=list[ConversationSummary])
def list_conversations(user: CurrentUserDep, session: SessionDep) -> list[Conversation]:
    statement = (
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(session.exec(statement))


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str, user: CurrentUserDep, session: SessionDep
) -> ConversationDetail:
    conversation = get_owned_conversation(conversation_id, user.id, session)
    messages = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    )
    return ConversationDetail(
        **conversation.model_dump(),
        messages=[MessageOut(**message.model_dump()) for message in messages],
    )


@router.patch("/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(
    conversation_id: str, request: ConversationUpdate, user: CurrentUserDep, session: SessionDep
) -> Conversation:
    conversation = get_owned_conversation(conversation_id, user.id, session)
    conversation.title = request.title
    conversation.updated_at = datetime.now(UTC)
    session.add(conversation)
    session.commit()
    session.refresh(conversation)
    return conversation


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, user: CurrentUserDep, session: SessionDep) -> None:
    conversation = get_owned_conversation(conversation_id, user.id, session)
    for message in session.exec(select(Message).where(Message.conversation_id == conversation.id)):
        session.delete(message)
    session.delete(conversation)
    session.commit()


def get_owned_conversation(conversation_id: str, user_id: str, session: Session) -> Conversation:
    """Shared with api/chat.py. Same 404 whether the conversation doesn't
    exist or belongs to someone else - not revealing which, on purpose."""
    conversation = session.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user_id:
        raise ConversationNotFoundError(conversation_id)
    return conversation
