"""The RAG engine: the only entry point the API, the CLI and the eval scripts use.

It knows nothing about HTTP. Components are injected, so a test can swap the LLM
or the store for a fake without touching this file.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from pydantic import BaseModel

from abstractrag.core.config import Settings
from abstractrag.core.logging import get_logger
from abstractrag.database.qdrant_store import QdrantStore
from abstractrag.rag.chunking.section_aware import SectionAwareChunker, embedding_text
from abstractrag.rag.embedding.bge_m3 import BgeM3Embedder
from abstractrag.rag.generation import prompts
from abstractrag.rag.generation.llm_client import LlamaCppClient
from abstractrag.rag.ingestion.base import SourceInput
from abstractrag.rag.ingestion.resolver import SourceResolver
from abstractrag.rag.models import (
    Answer,
    Chunk,
    Citation,
    IngestResult,
    RetrievedChunk,
    SectionSummary,
    VerifiedClaim,
)
from abstractrag.rag.reranking.cross_encoder import CrossEncoderReranker
from abstractrag.rag.retrieval.hybrid import HybridRetriever
from abstractrag.rag.summarization import prompts as summary_prompts
from abstractrag.rag.summarization.map_reduce import (
    MapReduceSummarizer,
    citations_for,
    passages_for,
)
from abstractrag.rag.verification.claims import split_claims
from abstractrag.rag.verification.verifier import ClaimVerifier

logger = get_logger(__name__)


class AnswerEvent(BaseModel):
    """One SSE event. The citations arrive before the first token so a UI can
    render the sources while the answer is still being written."""

    event: str  # citations | token | done
    citations: list[Citation] | None = None
    token: str | None = None
    answer: Answer | None = None


@dataclass
class _ContextSelection:
    """Retrieval output plus whether it clears the abstain threshold.

    Carries the chunks even when it does not clear the bar, so a failed question
    stays diagnosable: was the right section never retrieved, or was it retrieved
    and then rejected by the threshold? The two need different fixes.
    """

    chunks: list[RetrievedChunk]
    sufficient: bool


@dataclass
class RagEngine:
    settings: Settings
    resolver: SourceResolver
    chunker: SectionAwareChunker
    embedder: BgeM3Embedder
    store: QdrantStore
    retriever: HybridRetriever
    reranker: CrossEncoderReranker
    llm: LlamaCppClient
    verifier: ClaimVerifier
    summarizer: MapReduceSummarizer

    def ingest(self, source: SourceInput) -> IngestResult:
        document = self.resolver.resolve(source)
        chunks = self.chunker.chunk(document)
        logger.info("ingesting %s: %d chunks", document.title, len(chunks))

        self.store.ensure_collection()
        # Re-ingesting a source replaces it instead of duplicating it.
        self.store.delete_document(document.document_id)
        embeddings = self.embedder.embed([embedding_text(chunk) for chunk in chunks])
        self.store.upsert_chunks(chunks, embeddings)

        pages = [block.page for block in document.blocks if block.page is not None]
        return IngestResult(
            document_id=document.document_id,
            title=document.title,
            source_type=document.source_type,
            origin=document.origin,
            chunk_count=len(chunks),
            page_count=max(pages) if pages else None,
        )

    def answer(self, question: str, document_id: str | None = None) -> Answer:
        selection = self._select_context(question, document_id)
        if not selection.sufficient:
            return _abstain(selection.chunks)

        self._free_gpu_for_llm()
        context_chunks = selection.chunks
        # used_chunks keeps rank order (clearer for the debug UI); the LLM gets the
        # lost-in-the-middle order instead, and citation markers follow that order.
        context, citations = prompts.build_context(prompts.order_for_context(context_chunks))
        text = self.llm.complete(prompts.build_messages(question, context))
        if prompts.ABSTAIN_SENTINEL in text:
            return _abstain(context_chunks)

        return Answer(
            text=text,
            citations=_used_citations(text, citations),
            used_chunks=context_chunks,
            verified_claims=self._verify(text, citations, context_chunks),
        )

    def stream_answer(
        self, question: str, document_id: str | None = None
    ) -> Iterator[AnswerEvent]:
        selection = self._select_context(question, document_id)
        if not selection.sufficient:
            yield AnswerEvent(event="done", answer=_abstain(selection.chunks))
            return

        self._free_gpu_for_llm()
        context_chunks = selection.chunks
        context, citations = prompts.build_context(prompts.order_for_context(context_chunks))
        yield AnswerEvent(event="citations", citations=citations)

        collected = ""
        # Hold back the first tokens: the model signals "no answer here" with a
        # sentinel, and streaming half of it before noticing would look like an answer.
        held = True
        for token in self.llm.stream(prompts.build_messages(question, context)):
            collected += token
            if held:
                if len(collected) < len(prompts.ABSTAIN_SENTINEL):
                    continue
                if prompts.ABSTAIN_SENTINEL in collected:
                    yield AnswerEvent(event="done", answer=_abstain(context_chunks))
                    return
                held = False
                yield AnswerEvent(event="token", token=collected)
                continue
            yield AnswerEvent(event="token", token=token)

        if prompts.ABSTAIN_SENTINEL in collected:
            yield AnswerEvent(event="done", answer=_abstain(context_chunks))
            return

        text = collected.strip()
        yield AnswerEvent(
            event="done",
            answer=Answer(
                text=text,
                citations=_used_citations(text, citations),
                used_chunks=context_chunks,
                # Verification needs the whole answer, so it runs once the stream
                # has finished and rides along in this final event.
                verified_claims=self._verify(text, citations, context_chunks),
            ),
        )

    def summarize(self, question: str | None, document_id: str) -> Answer:
        """Answer a global question by summarising every section, not the top-k.

        "What is this paper about" has its answer spread across the document, so
        retrieving a handful of chunks cannot reach it. Costs one LLM call per
        section plus one - see rag.md 8.1.
        """
        chunks = self.store.list_chunks(document_id)
        if not chunks:
            return _abstain([])

        self._free_gpu_for_llm()
        text, summaries = self.summarizer.summarize(
            question or summary_prompts.DEFAULT_REQUEST, chunks
        )
        if not text:
            return _abstain([])

        return Answer(
            text=text,
            citations=citations_for(text, summaries, chunks),
            section_summaries=summaries,
            verified_claims=self._verify_summary(text, summaries, chunks),
        )

    def _verify_summary(
        self, text: str, summaries: list[SectionSummary], chunks: list[Chunk]
    ) -> list[VerifiedClaim]:
        """Same judge as an answer, but a marker resolves to the section's source
        text rather than one chunk - so a claim invented while summarising that
        section is caught instead of confirmed by its own summary."""
        if not self.settings.verification.enabled:
            return []
        return self.verifier.verify_passages(split_claims(text), passages_for(summaries, chunks))

    def preview_retrieval(
        self, question: str, document_id: str | None = None
    ) -> tuple[list[RetrievedChunk], bool]:
        """Retrieve + rerank without ever calling the LLM.

        For ablation runs that only need recall@k/MRR: generation is the slow
        part (tens of seconds per question), and retrieval metrics do not need
        an answer, just the ranked chunks. `sufficient` matches what `answer()`
        would have done - False means it would have abstained.
        """
        selection = self._select_context(question, document_id)
        return selection.chunks, selection.sufficient

    def _select_context(self, question: str, document_id: str | None) -> _ContextSelection:
        """Retrieve, rerank, and apply the abstain threshold.

        `sufficient=False` means the LLM is never asked a question its context
        cannot support - the cheapest hallucination guard.
        """
        candidates = self.retriever.retrieve(question, document_id)
        if not candidates:
            return _ContextSelection(chunks=[], sufficient=False)

        top = self.reranker.rerank(question, candidates, self.settings.retrieval.context_size)
        if not top or top[0].effective_score < self.settings.retrieval.score_threshold:
            logger.info("abstaining: best score below threshold")
            return _ContextSelection(chunks=top, sufficient=False)
        return _ContextSelection(chunks=top, sufficient=True)

    def _verify(
        self,
        text: str,
        citations: list[Citation],
        chunks: list[RetrievedChunk],
    ) -> list[VerifiedClaim]:
        """Check each sentence against the passage it cited.

        Only runs on answers that actually claimed something - an abstain never
        reaches here, because there is nothing to check.
        """
        if not self.settings.verification.enabled:
            return []
        return self.verifier.verify(split_claims(text), citations, chunks)

    def _free_gpu_for_llm(self) -> None:
        """Embedder and reranker shared the GPU with the LLM one at a time, never
        concurrently - free their VRAM right before an LLM call so it isn't
        fighting them for memory (measured: this was a 6x slowdown otherwise).

        Only called on the path that is about to call the LLM - preview_retrieval()
        has no LLM call to make room for, so it skips this and stays fast.
        """
        self.embedder.unload()
        self.reranker.unload()

    def health(self) -> dict[str, bool]:
        return {
            "qdrant": self.store.client.collection_exists(self.store.collection),
            "llm": self.llm.health(),
        }


def _abstain(used: list[RetrievedChunk] | None = None) -> Answer:
    return Answer(text=prompts.ABSTAIN_MESSAGE, abstained=True, used_chunks=used or [])


def _used_citations(text: str, citations: list[Citation]) -> list[Citation]:
    """Only return sources the answer actually pointed at."""
    markers = {int(marker) for marker in prompts.CITATION_MARKER.findall(text)}
    return [citation for citation in citations if citation.marker in markers]
