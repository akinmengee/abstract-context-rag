"""Multi-hop retrieval: search again for what the first pass could not reach.

"What negatives trained the retriever RAG uses?" needs two passages that no
single query retrieves together: RAG's paper says its retriever is DPR, and
only DPR's paper says how DPR was trained. The second query cannot be written
before the first passage is read, so the searches are planned one at a time
(self-ask style) rather than decomposed up front - which also covers
comparisons, where the planner simply asks for the side still missing.

Each search goes through the corrective agent, so every hop is graded; the
answer is then written once, from everything the hops kept.
"""

from dataclasses import dataclass

from abstractrag.core.config import AgentSettings
from abstractrag.core.logging import get_logger
from abstractrag.rag.agents import prompts
from abstractrag.rag.agents.base import ContextSelection, ReleaseGpu, Search
from abstractrag.rag.agents.corrective import CorrectiveAgent
from abstractrag.rag.generation.llm_client import LlamaCppClient
from abstractrag.rag.models import AgentStep, RetrievedChunk

logger = get_logger(__name__)


@dataclass
class MultiHopAgent:
    llm: LlamaCppClient
    settings: AgentSettings
    corrective: CorrectiveAgent

    def select(
        self,
        question: str,
        document_id: str | None,
        search: Search,
        release_gpu: ReleaseGpu,
    ) -> ContextSelection:
        first = self.corrective.select(question, document_id, search, release_gpu)
        steps: list[AgentStep] = list(first.steps)
        pool: list[RetrievedChunk] = list(first.chunks) if first.sufficient else []
        searches = [step.query for step in steps]

        planned = 0
        while len(searches) < self.settings.max_searches:
            release_gpu()
            messages = prompts.build_plan_messages(
                question, pool, searches, allow_done=planned > 0
            )
            planned += 1
            query = prompts.parse_plan(self.llm.complete(messages))
            if query is None or query.casefold() in {s.casefold() for s in searches}:
                break

            logger.info("follow-up search: %r", query)
            # Graded against the sub-query: that is what this hop's passages
            # have to answer, and a narrow question grades more reliably.
            hop = self.corrective.select(query, document_id, search, release_gpu)
            steps.extend(hop.steps)
            searches.extend(step.query for step in hop.steps)
            if hop.sufficient:
                pool = _merge(pool, hop.chunks, self.settings.max_context_chunks)

        return ContextSelection(chunks=pool, sufficient=bool(pool), steps=steps)


def _merge(
    pool: list[RetrievedChunk], new: list[RetrievedChunk], limit: int
) -> list[RetrievedChunk]:
    """Append unseen chunks in hop order, up to the context budget.

    Hop order rather than score order: a later hop's chunk is the second half
    of the answer, and ranking it against the first hop's would drop it.
    """
    seen = {chunk.chunk.chunk_id for chunk in pool}
    merged = list(pool)
    for chunk in new:
        if len(merged) >= limit:
            break
        if chunk.chunk.chunk_id not in seen:
            merged.append(chunk)
            seen.add(chunk.chunk.chunk_id)
    return merged
