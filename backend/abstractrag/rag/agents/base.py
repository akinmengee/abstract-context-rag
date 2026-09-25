"""What an agent hands back to the engine, and the tools the engine lends it.

The engine owns retrieval and the GPU; an agent only decides what to search for
and what to keep. Passing `search` and `release_gpu` in per call keeps agents
free of any engine import and trivially fakeable in tests.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from abstractrag.rag.models import AgentStep, RetrievedChunk

# (query, document_id) -> reranked top chunks, no threshold applied.
Search = Callable[[str, str | None], list[RetrievedChunk]]
# Frees the embedder/reranker VRAM before an LLM call.
ReleaseGpu = Callable[[], None]


@dataclass
class ContextSelection:
    """Chunks for the LLM plus whether they are good enough to answer from.

    Carries the chunks even when they are not, so a failed question stays
    diagnosable: was the right section never retrieved, or was it retrieved and
    then rejected? The two need different fixes.
    """

    chunks: list[RetrievedChunk]
    sufficient: bool
    steps: list[AgentStep] = field(default_factory=list)
