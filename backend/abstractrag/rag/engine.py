"""The RAG engine: the only entry point the API, the CLI and the eval scripts use.

It knows nothing about HTTP. Components are injected, so a test can swap the LLM
or the store for a fake without touching this file.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from abstractrag.core.config import Settings
from abstractrag.core.logging import get_logger
from abstractrag.database.qdrant_store import QdrantStore
from abstractrag.rag.agents.base import ContextSelection
from abstractrag.rag.agents.corrective import CorrectiveAgent
from abstractrag.rag.agents.multi_hop import MultiHopAgent
from abstractrag.rag.chunking.section_aware import SectionAwareChunker, embedding_text
from abstractrag.rag.embedding.bge_m3 import BgeM3Embedder
from abstractrag.rag.generation import prompts
from abstractrag.rag.generation.llm_client import LlamaCppClient
from abstractrag.rag.generation.vision import VisionDescriber
from abstractrag.rag.ingestion.base import SourceInput
from abstractrag.rag.ingestion.resolver import SourceResolver
from abstractrag.rag.models import (
    AgentStep,
    Answer,
    Chunk,
    Citation,
    IngestResult,
    RetrievedChunk,
    SectionSummary,
    VerifiedClaim,
)
from abstractrag.rag.query.router import is_global_question
from abstractrag.rag.raptor.tree import TreeBuilder
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
    # None is the plain single-pass pipeline (agent.mode = off).
    agent: CorrectiveAgent | MultiHopAgent | None = None
    # None until wired: a RAPTOR tree is optional per document (rag.md 7.9.1).
    tree_builder: TreeBuilder | None = None
    # None unless ingestion.figures and vision are both enabled (rag.md 7.9.4):
    # describes a retrieved figure with the user's real question instead of a
    # fixed ingest-time caption.
    vision: VisionDescriber | None = None

    def ingest(self, source: SourceInput) -> IngestResult:
        document = self.resolver.resolve(source)
        chunks = self.chunker.chunk(document)
        logger.info("ingesting %s: %d chunks", document.title, len(chunks))

        self.store.ensure_collection()
        # Re-ingesting a source replaces it instead of duplicating it.
        self.store.delete_document(document.document_id)
        embeddings = self.embedder.embed([embedding_text(chunk) for chunk in chunks])
        self.store.upsert_chunks(chunks, embeddings)

        tree_nodes = (
            self.build_tree(document.document_id) if self.settings.raptor.build_on_ingest else 0
        )
        pages = [block.page for block in document.blocks if block.page is not None]
        return IngestResult(
            document_id=document.document_id,
            title=document.title,
            source_type=document.source_type,
            origin=document.origin,
            chunk_count=len(chunks),
            page_count=max(pages) if pages else None,
            tree_nodes=tree_nodes,
        )

    def build_tree(self, document_id: str) -> int:
        """(Re)build one document's RAPTOR tree; returns the number of nodes."""
        leaves = self.store.list_chunks(document_id)
        if not leaves:
            return 0
        self.store.delete_tree(document_id)
        built = self.tree_builder.build(leaves, self._free_gpu_for_llm)
        if built:
            self.store.upsert_chunks([node for node, _ in built], [e for _, e in built])
        return len(built)

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Answer:
        """Route a question to retrieval or map-reduce summarisation.

        A specific question is answered from the top-k; a global one
        ("summarize this paper") cannot be - top-k structurally misses most of
        the answer (rag.md 8, 7.4.1). `summarize()` needs one document to scope
        to, so a global-sounding question with no document_id still falls
        through to answer() rather than guessing which document to summarise.

        `history` is recent prior turns of the same conversation (rag.md 12) -
        never passed to summarize(), a whole-document operation with no
        conversational framing.

        This is what `/api/v1/chat` and `abstractrag ask` call by default.
        Callers that want one specific path regardless of phrasing use
        answer() or summarize() directly - `/api/v1/summarize` and
        `abstractrag summarize` are unrouted on purpose.
        """
        if document_id and is_global_question(question):
            return self.summarize(question, document_id)
        return self.answer(question, document_id, history)

    def answer(
        self,
        question: str,
        document_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Answer:
        selection = self._select_context(question, document_id)
        if not selection.sufficient:
            return self._try_summary_fallback(question, document_id) or _abstain(
                selection.chunks, selection.steps
            )

        self._free_gpu_for_llm()
        context_chunks = self._augment_figures(question, selection.chunks)
        # used_chunks keeps rank order (clearer for the debug UI); the LLM gets the
        # lost-in-the-middle order instead, and citation markers follow that order.
        context, citations = prompts.build_context(prompts.order_for_context(context_chunks))
        text = self.llm.complete(prompts.build_messages(question, context, history))
        if prompts.ABSTAIN_SENTINEL in text:
            return self._try_summary_fallback(question, document_id) or _abstain(
                context_chunks, selection.steps
            )

        return Answer(
            text=text,
            citations=_used_citations(text, citations),
            used_chunks=context_chunks,
            verified_claims=self._verify(text, citations, context_chunks),
            agent_steps=selection.steps,
        )

    def stream_answer(
        self,
        question: str,
        document_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[AnswerEvent]:
        selection = self._select_context(question, document_id)
        if not selection.sufficient:
            fallback = self._try_summary_fallback(question, document_id)
            if fallback is not None:
                yield from answer_as_events(fallback)
                return
            yield AnswerEvent(event="done", answer=_abstain(selection.chunks, selection.steps))
            return

        self._free_gpu_for_llm()
        context_chunks = self._augment_figures(question, selection.chunks)
        context, citations = prompts.build_context(prompts.order_for_context(context_chunks))
        yield AnswerEvent(event="citations", citations=citations)

        collected = ""
        # Hold back the first tokens: the model signals "no answer here" with a
        # sentinel, and streaming half of it before noticing would look like an answer.
        held = True
        for token in self.llm.stream(prompts.build_messages(question, context, history)):
            collected += token
            if held:
                if len(collected) < len(prompts.ABSTAIN_SENTINEL):
                    continue
                if prompts.ABSTAIN_SENTINEL in collected:
                    # Nothing has reached the client yet (still held), so a
                    # fallback can still swap in a whole different answer
                    # cleanly - unlike the same check after the loop below,
                    # where token events may already be showing.
                    fallback = self._try_summary_fallback(question, document_id)
                    if fallback is not None:
                        yield from answer_as_events(
                            fallback, include_citations=False
                        )
                        return
                    yield AnswerEvent(
                        event="done", answer=_abstain(context_chunks, selection.steps)
                    )
                    return
                held = False
                yield AnswerEvent(event="token", token=collected)
                continue
            yield AnswerEvent(event="token", token=token)

        if prompts.ABSTAIN_SENTINEL in collected:
            # No fallback here (unlike the same check above, still inside
            # `held`): token events for this answer have already reached the
            # client, so swapping in a whole different one now would just
            # append a second, unrelated answer after visible wrong content.
            yield AnswerEvent(event="done", answer=_abstain(context_chunks, selection.steps))
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
                agent_steps=selection.steps,
            ),
        )

    def summarize(self, question: str | None, document_id: str) -> Answer:
        """Answer a global question by summarising every section, not the top-k.

        "What is this paper about" has its answer spread across the document, so
        retrieving a handful of chunks cannot reach it. map_reduce costs one LLM
        call per section plus one (rag.md 8.1); raptor reduces the document's
        precomputed tree nodes in one call (rag.md 7.9.1).
        """
        chunks = self.store.list_chunks(document_id)
        if not chunks:
            return _abstain([])

        self._free_gpu_for_llm()
        request = question or summary_prompts.DEFAULT_REQUEST
        method = self.settings.summarization.method
        summaries = self._tree_summaries(document_id) if method == "raptor" else []
        if summaries:
            text = self.summarizer.reduce(request, summaries)
        else:
            if method == "raptor":
                logger.warning("no RAPTOR tree for %s, summarising with map-reduce", document_id)
            text, summaries = self.summarizer.summarize(request, chunks)
        if not text:
            return _abstain([])

        return Answer(
            text=text,
            citations=citations_for(text, summaries, chunks),
            section_summaries=summaries,
            verified_claims=self._verify_summary(text, summaries, chunks),
        )

    def _try_summary_fallback(self, question: str, document_id: str | None) -> Answer | None:
        """A safety net for a global question the keyword router missed.

        is_global_question() is a fixed list of English trigger words on
        purpose (rag.md 7.4.1) - cheap, but never exhaustive. Measured live:
        "what is this research about" fell through to plain retrieval for a
        Wikipedia article, because "research" wasn't a recognised document
        word - retrieval abstained, when summarize() would have answered it.
        Rather than only ever closing this gap by chasing more synonyms,
        retrying here catches whatever the router still misses, whenever
        retrieval itself gives up - at the cost of a slower genuine abstain,
        which now pays for this attempt too before giving up for real.

        Returns None (never a fallback answer) when there's no single
        document to scope to, or when summarize() itself abstains too - the
        caller falls back to its own abstain in either case.
        """
        if document_id is None:
            return None
        fallback = self.summarize(question, document_id)
        return None if fallback.abstained else fallback

    def _tree_summaries(self, document_id: str) -> list[SectionSummary]:
        """The level-1 tree nodes as numbered section summaries.

        `chunk_ids` are the leaves each node covers, so citations_for() and
        passages_for() resolve a marker to real source text exactly as they do
        for map-reduce.
        """
        return [
            SectionSummary(
                marker=number,
                section=node.metadata.section or node.metadata.title,
                text=node.text,
                chunk_ids=node.metadata.source_ids,
            )
            for number, node in enumerate(self.store.list_chunks(document_id, level=1), start=1)
        ]

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
        """Choose the context without generating an answer.

        For ablation runs that only need retrieval metrics: generation is the
        slow part (tens of seconds per question). With agent.mode off this never
        calls the LLM; an agent still needs it to grade and plan. `sufficient`
        matches what `answer()` would have done - False means it would have
        abstained.
        """
        selection = self._select_context(question, document_id)
        return selection.chunks, selection.sufficient

    def _select_context(self, question: str, document_id: str | None) -> ContextSelection:
        """Retrieve, rerank and apply the abstain threshold - or let the agent choose.

        `sufficient=False` means the LLM is never asked a question its context
        cannot support - the cheapest hallucination guard.
        """
        if self.agent is not None:
            return self.agent.select(question, document_id, self._search, self._free_gpu_for_llm)

        top = self._search(question, document_id)
        if not top or top[0].effective_score < self.settings.retrieval.score_threshold:
            logger.info("abstaining: best score below threshold")
            return ContextSelection(chunks=top, sufficient=False)
        return ContextSelection(chunks=top, sufficient=True)

    def _search(self, query: str, document_id: str | None) -> list[RetrievedChunk]:
        """Stage 1 + stage 2 retrieval, no threshold: the top chunks for a query."""
        candidates = self.retriever.retrieve(query, document_id)
        if not candidates:
            return []
        return self.reranker.rerank(query, candidates, self.settings.retrieval.context_size)

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
        claims = split_claims(text)
        if not any(chunk.chunk.metadata.level for chunk in chunks):
            return self.verifier.verify(claims, citations, chunks)
        return self.verifier.verify_passages(claims, self._source_passages(citations, chunks))

    def _source_passages(
        self, citations: list[Citation], chunks: list[RetrievedChunk]
    ) -> dict[int, str]:
        """Marker -> source text, resolving a cited tree node to its leaves.

        A node's own text is a summary; checking a claim against it would let
        a sentence invented while summarising confirm itself (rag.md 8.1).
        """
        by_id = {chunk.chunk.chunk_id: chunk.chunk for chunk in chunks}
        nodes = [chunk for chunk in by_id.values() if chunk.metadata.level]
        leaf_ids = sorted({leaf for node in nodes for leaf in node.metadata.source_ids})
        leaves = {leaf.chunk_id: leaf.text for leaf in self.store.get_chunks(leaf_ids)}
        passages: dict[int, str] = {}
        for citation in citations:
            chunk = by_id.get(citation.chunk_id)
            if chunk is None:
                continue
            passages[citation.marker] = (
                "\n\n".join(leaves[i] for i in chunk.metadata.source_ids if i in leaves)
                if chunk.metadata.level
                else chunk.text
            )
        return passages

    def _augment_figures(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Swap a figure chunk's text for a fresh answer to this exact question
        (rag.md 7.9.4) - never written back to the store, this query only.
        Falls back to the chunk's existing text if vision is off or the call
        failed, so a bad figure lookup never turns into a missing chunk."""
        if self.vision is None:
            return chunks

        augmented: list[RetrievedChunk] = []
        for retrieved in chunks:
            paths = retrieved.chunk.metadata.image_paths
            if not paths:
                augmented.append(retrieved)
                continue
            descriptions = [
                text
                for path in paths
                if (text := self.vision.describe(Path(path), question))
            ]
            if not descriptions:
                augmented.append(retrieved)
                continue
            new_chunk = retrieved.chunk.model_copy(update={"text": "\n\n".join(descriptions)})
            augmented.append(retrieved.model_copy(update={"chunk": new_chunk}))
        return augmented

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


def _abstain(
    used: list[RetrievedChunk] | None = None, steps: list[AgentStep] | None = None
) -> Answer:
    return Answer(
        text=prompts.ABSTAIN_MESSAGE,
        abstained=True,
        used_chunks=used or [],
        agent_steps=steps or [],
    )


def _used_citations(text: str, citations: list[Citation]) -> list[Citation]:
    """Only return sources the answer actually pointed at."""
    markers = {int(marker) for marker in prompts.CITATION_MARKER.findall(text)}
    return [citation for citation in citations if citation.marker in markers]


def answer_as_events(
    answer: Answer, include_citations: bool = True
) -> Iterator[AnswerEvent]:
    """Turn an already-computed Answer into the same event shape
    stream_answer() produces token-by-token: summarize() (used directly by
    api/chat.py for a router-detected global question, and by
    RagEngine._try_summary_fallback() when retrieval itself gives up) has no
    incremental tokens to stream - the whole computation finishes before
    there's any text at all - so the answer goes out as one `token` event
    instead of many.

    `include_citations=False` when a `citations` event for this turn has
    already been sent (RagEngine.stream_answer()'s mid-stream fallback case)
    - the contract is one `citations` event per turn, not per answer tried.
    """
    if include_citations:
        yield AnswerEvent(event="citations", citations=answer.citations)
    if not answer.abstained:
        yield AnswerEvent(event="token", token=answer.text)
    yield AnswerEvent(event="done", answer=answer)
