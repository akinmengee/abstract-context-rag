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
    """Answer a question from the indexed sources, or abstain if they cannot.

    Routes global questions ("summarize this paper") to map-reduce
    summarisation instead - see RagEngine.ask().
    """
    return engine.ask(request.question, request.document_id)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest, engine: EngineDep) -> StreamingResponse:
    """Answer as Server-Sent Events - but never routed to summarisation.

    Unlike /chat, this always calls stream_answer() directly, which is
    retrieval-only: a global question ("summarize this paper") streams a
    normal (and likely abstained or weak) retrieval answer here, where /chat
    would instead route it to map-reduce summarisation. So for the same
    global question, this endpoint's answer can differ from /chat's.

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
    """The same summarisation /chat routes to automatically, called directly.

    Bypasses routing - no question-text check, just document_id in, summary
    out. For explicit or scripted use when the caller already knows it wants
    a full-document summary; /chat can take just as long when it routes a
    global question here on its own.
    """
    return engine.summarize(request.question, request.document_id)


def _sse(engine: RagEngine, request: ChatRequest) -> Iterator[str]:
    for event in engine.stream_answer(request.question, request.document_id):
        payload = event.model_dump_json(exclude_none=True)
        yield f"event: {event.event}\ndata: {payload}\n\n"
