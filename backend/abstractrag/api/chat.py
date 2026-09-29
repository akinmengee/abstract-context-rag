"""Question answering endpoints, blocking and streaming - now
conversation-scoped: every question is asked inside an existing
conversation, which fixes which document(s) it can draw on and gives the
engine a little conversational memory."""

from collections.abc import Iterator
from datetime import UTC, datetime

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy import Engine
from sqlmodel import Session, select

from abstractrag.accounts.models import Conversation, Message, MessageRole
from abstractrag.api.conversations import get_owned_conversation
from abstractrag.api.dependencies import CurrentUserDep, DbEngineDep, EngineDep, SessionDep
from abstractrag.api.schemas import ChatRequest, SummarizeRequest
from abstractrag.rag.engine import AnswerEvent, RagEngine, answer_as_events
from abstractrag.rag.models import Answer
from abstractrag.rag.query.router import is_global_question

router = APIRouter(tags=["chat"])

# Last 3 user/assistant turns (6 messages), spliced ahead of the current
# question - enough for pronoun/follow-up resolution ("so how does this
# work") without re-retrieval or query rewriting (rag.md 12).
HISTORY_TURNS = 3


@router.post("/chat", response_model=Answer)
def chat(
    request: ChatRequest, engine: EngineDep, user: CurrentUserDep, session: SessionDep
) -> Answer:
    """Answer a question from the indexed sources, or abstain if they cannot.

    Routes global questions ("summarize this paper") to map-reduce
    summarisation instead - see RagEngine.ask(). The conversation fixes
    which document(s) this can draw on; both turns are persisted to it.
    """
    conversation = get_owned_conversation(request.conversation_id, user.id, session)
    history = _recent_history(session, conversation.id)
    _append_message(session, conversation, MessageRole.USER, request.question, citations=None)

    answer = engine.ask(request.question, conversation.document_id, history=history)

    _append_message(
        session,
        conversation,
        MessageRole.ASSISTANT,
        answer.text,
        citations=[c.model_dump() for c in answer.citations] or None,
    )
    return answer


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    engine: EngineDep,
    user: CurrentUserDep,
    session: SessionDep,
    db_engine: DbEngineDep,
) -> StreamingResponse:
    """Answer as Server-Sent Events, routed the same way /chat is.

    A global question ("summarize this paper") has no map-reduce/RAPTOR
    tokens to stream incrementally - summarize() computes the whole answer
    before anything can be shown - so that case sends its citations and full
    text as a single `token` event instead of the usual token-by-token
    stream, then `done`. Everything else streams normally via
    stream_answer(). Same event order either way: `citations` once, then one
    or more `token` events, then `done` carrying the final answer (including
    the abstain flag). An abstain sends `done` without any tokens.
    """
    conversation = get_owned_conversation(request.conversation_id, user.id, session)
    history = _recent_history(session, conversation.id)
    _append_message(session, conversation, MessageRole.USER, request.question, citations=None)

    if conversation.document_id and is_global_question(request.question):
        events = _sse_summary(engine, request.question, conversation.document_id)
    else:
        events = _sse_stream(engine, request.question, conversation.document_id, history)

    return StreamingResponse(
        _persist_final_answer(events, conversation.id, db_engine),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/summarize", response_model=Answer)
def summarize(request: SummarizeRequest, engine: EngineDep) -> Answer:
    """The same summarisation /chat routes to automatically, called directly.

    Bypasses routing - no question-text check, just document_id in, summary
    out. Not conversation-scoped: a whole-document operation, not a chat turn.
    """
    return engine.summarize(request.question, request.document_id)


def _sse_stream(
    engine: RagEngine, question: str, document_id: str | None, history: list[dict[str, str]]
) -> Iterator[AnswerEvent]:
    yield from engine.stream_answer(question, document_id, history=history)


def _sse_summary(engine: RagEngine, question: str, document_id: str) -> Iterator[AnswerEvent]:
    return answer_as_events(engine.summarize(question, document_id))


def _persist_final_answer(
    events: Iterator[AnswerEvent], conversation_id: str, db_engine: Engine
) -> Iterator[str]:
    final_answer: Answer | None = None
    for event in events:
        if event.event == "done":
            final_answer = event.answer
        yield f"event: {event.event}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"

    # A fresh session here, not the request-scoped SessionDep: this generator
    # keeps running after the route handler itself has returned, and reusing
    # a yield-dependency session across that boundary depends on FastAPI/
    # Starlette internals this module shouldn't have to reason about. The
    # Engine itself still comes through Depends() (DbEngineDep), not a direct
    # get_db_engine() call, so tests can override it same as SessionDep.
    if final_answer is not None:
        with Session(db_engine) as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                _append_message(
                    session,
                    conversation,
                    MessageRole.ASSISTANT,
                    final_answer.text,
                    citations=[c.model_dump() for c in final_answer.citations] or None,
                )


def _recent_history(session: Session, conversation_id: str) -> list[dict[str, str]]:
    statement = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(HISTORY_TURNS * 2)
    )
    recent = list(session.exec(statement))[::-1]  # back to chronological order
    return [{"role": message.role.value, "content": message.content} for message in recent]


def _append_message(
    session: Session,
    conversation: Conversation,
    role: MessageRole,
    content: str,
    citations: list[dict] | None,
) -> None:
    message = Message(
        conversation_id=conversation.id, role=role, content=content, citations=citations
    )
    session.add(message)
    conversation.updated_at = datetime.now(UTC)
    session.add(conversation)
    session.commit()
