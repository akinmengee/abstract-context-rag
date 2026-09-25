"""Corrective retrieval: check what came back before answering from it.

The reranker threshold (rag.md 9) is cheap, but it scores topical similarity:
"What MRR@10 does RAPTOR achieve on MS MARCO?" scores 0.77 against ColBERT's
MS MARCO results, because nothing in a relevance score says "wrong system". An
LLM grader reads question and passages together and decides whether any of
them answers it. When none does - or the threshold already said so - the query
is rewritten and searched once more before giving up.

Cost: one grading call per search, plus a rewrite and a second grading call
when the first search misses.
"""

from dataclasses import dataclass

from abstractrag.core.config import AgentSettings
from abstractrag.core.logging import get_logger
from abstractrag.rag.agents import prompts
from abstractrag.rag.agents.base import ContextSelection, ReleaseGpu, Search
from abstractrag.rag.generation.llm_client import LlamaCppClient
from abstractrag.rag.models import AgentStep, RetrievedChunk

logger = get_logger(__name__)


@dataclass
class CorrectiveAgent:
    llm: LlamaCppClient
    settings: AgentSettings
    score_threshold: float
    # False: the grade is only a gate - if anything answers the question, the
    # whole retrieved context is kept. Filtering measurably dropped chunks an
    # answer needed (one side of a comparison), so it is only worth it where
    # several searches must share one context budget (multi-hop).
    filter_chunks: bool = False

    def select(
        self,
        question: str,
        document_id: str | None,
        search: Search,
        release_gpu: ReleaseGpu,
    ) -> ContextSelection:
        steps: list[AgentStep] = []
        query = question
        chunks: list[RetrievedChunk] = []

        for attempt in range(self.settings.max_rewrites + 1):
            chunks = search(query, document_id)
            kept = self._grade(question, chunks, release_gpu)
            steps.append(AgentStep(query=query, retrieved=len(chunks), kept=len(kept)))
            if kept:
                context = kept if self.filter_chunks else chunks
                return ContextSelection(chunks=context, sufficient=True, steps=steps)
            if attempt == self.settings.max_rewrites:
                break

            release_gpu()
            rewritten = prompts.parse_rewrite(
                self.llm.complete(prompts.build_rewrite_messages(question))
            )
            if not rewritten or rewritten.casefold() == query.casefold():
                break
            logger.info("nothing usable for %r; retrying as %r", query, rewritten)
            query = rewritten

        return ContextSelection(chunks=chunks, sufficient=False, steps=steps)

    def _grade(
        self, question: str, chunks: list[RetrievedChunk], release_gpu: ReleaseGpu
    ) -> list[RetrievedChunk]:
        """The chunks that answer the question, in rank order.

        Below the reranker threshold nothing is sent to the grader: that
        search already failed, and the call would only confirm it.
        """
        if not chunks or chunks[0].effective_score < self.score_threshold:
            return []

        release_gpu()
        reply = self.llm.complete(prompts.build_grade_messages(question, chunks))
        numbers = prompts.parse_grade(reply, len(chunks))
        if numbers is None:
            # A filter, not a verdict: an unreadable grade falls back to the
            # plain pipeline (keep what cleared the threshold), never below it.
            logger.warning("unreadable grade, keeping all chunks: %r", reply[:120])
            return chunks
        return [chunk for number, chunk in enumerate(chunks, start=1) if number in numbers]
