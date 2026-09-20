"""The RAG engine: the only entry point the API, the CLI and the eval scripts use.

It knows nothing about HTTP. Components are injected, so a test can swap the LLM
or the store for a fake without touching this file.
"""

import re
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
from abstractrag.rag.models import Answer, Citation, IngestResult, RetrievedChunk
from abstractrag.rag.reranking.cross_encoder import CrossEncoderReranker
from abstractrag.rag.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)

_CITATION_MARKER = re.compile(r"\[(\d+)\]")


class AnswerEvent(BaseModel):
    """One SSE event. The citations arrive before the first token so a UI can
    render the sources while the answer is still being written."""

    event: str  # citations | token | done
    citations: list[Citation] | None = None
    token: str | None = None
    answer: Answer | None = None


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
        context_chunks = self._select_context(question, document_id)
        if context_chunks is None:
            return _abstain()

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
        )

    def stream_answer(
        self, question: str, document_id: str | None = None
    ) -> Iterator[AnswerEvent]:
        context_chunks = self._select_context(question, document_id)
        if context_chunks is None:
            yield AnswerEvent(event="done", answer=_abstain())
            return

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

        yield AnswerEvent(
            event="done",
            answer=Answer(
                text=collected.strip(),
                citations=_used_citations(collected, citations),
                used_chunks=context_chunks,
            ),
        )

    def _select_context(
        self, question: str, document_id: str | None
    ) -> list[RetrievedChunk] | None:
        """Retrieve, rerank, and apply the abstain threshold.

        Returns None when the source clearly has no answer, so the LLM is never
        asked a question its context cannot support - the cheapest hallucination guard.
        """
        candidates = self.retriever.retrieve(question, document_id)
        if not candidates:
            return None

        top = self.reranker.rerank(question, candidates, self.settings.retrieval.context_size)
        if not top or top[0].effective_score < self.settings.retrieval.score_threshold:
            logger.info("abstaining: best score below threshold")
            return None

        # Embedder and reranker shared the GPU with the LLM one at a time, never
        # concurrently - free their VRAM now so the LLM call isn't fighting them
        # for memory (measured: this was a 6x slowdown on the LLM call otherwise).
        self.embedder.unload()
        self.reranker.unload()
        return top

    def health(self) -> dict[str, bool]:
        return {
            "qdrant": self.store.client.collection_exists(self.store.collection),
            "llm": self.llm.health(),
        }


def _abstain(used: list[RetrievedChunk] | None = None) -> Answer:
    return Answer(text=prompts.ABSTAIN_MESSAGE, abstained=True, used_chunks=used or [])


def _used_citations(text: str, citations: list[Citation]) -> list[Citation]:
    """Only return sources the answer actually pointed at."""
    markers = {int(marker) for marker in _CITATION_MARKER.findall(text)}
    return [citation for citation in citations if citation.marker in markers]
