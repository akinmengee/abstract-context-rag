"""Question answering endpoints, blocking and streaming."""

from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from abstractrag.api.dependencies import EngineDep
from abstractrag.api.schemas import ChatRequest, SummarizeRequest
from abstractrag.rag.engine import RagEngine
from abstractrag.rag.models import Answer

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=Answer)
def chat(request: ChatRequest, engine: EngineDep) -> Answer:
    """Answer a question from the indexed sources, or abstain if they cannot."""
    return engine.answer(request.question, request.document_id)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest, engine: EngineDep) -> StreamingResponse:
    """Same answer as /chat, streamed as Server-Sent Events.

    Event order is fixed: `citations` once, then `token` repeatedly, then `done`
    carrying the final answer (including the abstain flag). An abstain sends
    `done` without any tokens.
    """
    return StreamingResponse(
        _sse(engine, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/summarize", response_model=Answer)
def summarize(request: SummarizeRequest, engine: EngineDep) -> Answer:
    """Global questions and full summaries. Far slower than /chat by design."""
    return engine.summarize(request.question, request.document_id)


def _sse(engine: RagEngine, request: ChatRequest) -> Iterator[str]:
    for event in engine.stream_answer(request.question, request.document_id):
        payload = event.model_dump_json(exclude_none=True)
        yield f"event: {event.event}\ndata: {payload}\n\n"
