"""Engine behaviour with fake components: abstain rules, citations, streaming."""

from collections.abc import Iterator

from abstractrag.core.config import (
    RetrievalSettings,
    Settings,
    SummarizationSettings,
    VerificationSettings,
)
from abstractrag.rag.engine import RagEngine
from abstractrag.rag.generation.prompts import ABSTAIN_MESSAGE, ABSTAIN_SENTINEL
from abstractrag.rag.models import (
    ClaimVerdict,
    RetrievedChunk,
    SectionSummary,
    VerifiedClaim,
    node_id_for,
)
from tests.conftest import make_chunk


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def retrieve(self, query: str, document_id=None, limit=None) -> list[RetrievedChunk]:
        return list(self.results)


class FakeReranker:
    """Assigns a fixed score to every candidate, in the order they arrive."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.unloaded = False

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int):
        for candidate, score in zip(candidates, self.scores, strict=False):
            candidate.rerank_score = score
        candidates.sort(key=lambda candidate: candidate.effective_score, reverse=True)
        return candidates[:top_k]

    def unload(self) -> None:
        self.unloaded = True


class FakeEmbedder:
    def __init__(self) -> None:
        self.unloaded = False

    def unload(self) -> None:
        self.unloaded = True


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.last_messages: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.last_messages = messages
        return self.response

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        self.calls += 1
        self.last_messages = messages
        for index in range(0, len(self.response), 4):
            yield self.response[index : index + 4]


class FakeVerifier:
    """Marks every cited claim supported; counts calls so tests can check it ran."""

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, claims, citations, chunks) -> list[VerifiedClaim]:
        self.calls += 1
        return [
            VerifiedClaim(
                text=claim.text,
                markers=claim.markers,
                verdict=ClaimVerdict.SUPPORTED if claim.markers else ClaimVerdict.UNCITED,
            )
            for claim in claims
        ]

    def verify_passages(self, claims, passages) -> list[VerifiedClaim]:
        self.calls += 1
        self.last_passages = passages
        return [
            VerifiedClaim(
                text=claim.text,
                markers=claim.markers,
                verdict=ClaimVerdict.SUPPORTED if claim.markers else ClaimVerdict.UNCITED,
            )
            for claim in claims
        ]


class FakeVisionDescriber:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple] = []

    def describe(self, image_path, question: str) -> str:
        self.calls.append((image_path, question))
        return self.response


class FakeStore:
    def __init__(self, chunks: list = None, nodes: dict | None = None) -> None:
        self.chunks = chunks or []
        self.nodes = nodes or {}
        self.deleted_trees: list[str] = []
        self.upserted: list = []
        self.lookups: list[list[str]] = []

    def list_chunks(self, document_id: str, level: int = 0) -> list:
        return list(self.chunks) if level == 0 else list(self.nodes.get(level, []))

    def delete_tree(self, document_id: str) -> None:
        self.deleted_trees.append(document_id)

    def upsert_chunks(self, chunks, embeddings) -> None:
        self.upserted.extend(chunks)

    def get_chunks(self, chunk_ids: list[str]) -> list:
        self.lookups.append(list(chunk_ids))
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        return [by_id[i] for i in chunk_ids if i in by_id]


class FakeSummarizer:
    """Returns a canned (text, summaries) pair; records what it was asked."""

    def __init__(self, text: str, summaries: list[SectionSummary]) -> None:
        self.text = text
        self.summaries = summaries
        self.calls: list[tuple] = []

    def summarize(self, question, chunks):
        self.calls.append((question, chunks))
        return self.text, self.summaries

    def reduce(self, question, summaries):
        self.reduced = (question, list(summaries))
        return self.text


class FakeTreeBuilder:
    def __init__(self, nodes: list) -> None:
        self.nodes = nodes
        self.built_from: list = []

    def build(self, leaves, release_gpu):
        self.built_from.append(list(leaves))
        release_gpu()
        return [(node, None) for node in self.nodes]


def build_engine(
    candidates: list[RetrievedChunk],
    scores: list[float],
    response: str,
    verification_enabled: bool = True,
    store=None,
    summarizer=None,
    tree_builder=None,
    summary_method: str = "map_reduce",
    vision=None,
) -> RagEngine:
    settings = Settings(
        retrieval=RetrievalSettings(context_size=3, score_threshold=0.3),
        verification=VerificationSettings(enabled=verification_enabled),
        summarization=SummarizationSettings(method=summary_method),
    )
    return RagEngine(
        settings=settings,
        resolver=None,
        chunker=None,
        embedder=FakeEmbedder(),
        store=store,
        retriever=FakeRetriever(candidates),
        reranker=FakeReranker(scores),
        llm=FakeLLM(response),
        verifier=FakeVerifier(),
        summarizer=summarizer,
        tree_builder=tree_builder,
        vision=vision,
    )


def candidate(
    text: str, index: int, score: float = 0.5, image_paths: list[str] | None = None
) -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(text, index=index, image_paths=image_paths), score=score)


def test_abstains_without_asking_the_llm_when_nothing_is_retrieved():
    engine = build_engine([], [], "should never be used")

    answer = engine.answer("what is the learning rate?")

    assert answer.abstained
    assert answer.text == ABSTAIN_MESSAGE
    assert engine.llm.calls == 0


def test_abstains_without_asking_the_llm_when_scores_are_below_threshold():
    engine = build_engine([candidate("weak", 0)], [0.05], "should never be used")

    answer = engine.answer("unrelated question")

    assert answer.abstained
    assert engine.llm.calls == 0


def test_abstains_when_the_model_reports_the_source_cannot_answer():
    engine = build_engine([candidate("relevant", 0)], [0.9], ABSTAIN_SENTINEL)

    answer = engine.answer("something the paper never says")

    assert answer.abstained
    assert answer.text == ABSTAIN_MESSAGE


def test_only_referenced_sources_are_returned_as_citations():
    candidates = [candidate("first", 0), candidate("second", 1), candidate("third", 2)]
    engine = build_engine(candidates, [0.9, 0.8, 0.7], "The answer is 42 [1].")

    answer = engine.answer("what is the answer?")

    assert not answer.abstained
    assert [citation.marker for citation in answer.citations] == [1]
    assert len(answer.used_chunks) == 3


def test_context_uses_lost_in_the_middle_order_not_raw_rank_order():
    # Pages double as an index here: page N came from candidate N (see make_chunk).
    candidates = [candidate("first", 0), candidate("second", 1), candidate("third", 2)]
    engine = build_engine(candidates, [0.9, 0.8, 0.7], "Combines all three [1][2][3].")

    answer = engine.answer("question")

    # Rank order would be [1, 2, 3]; lost-in-the-middle puts rank 2 (page 3) before
    # rank 1 (page 2), since the best and third-best flank the weaker middle chunk.
    assert [citation.page for citation in answer.citations] == [1, 3, 2]
    # used_chunks stays in plain rank order - it feeds the debug UI, not the prompt.
    assert [chunk.chunk.metadata.page for chunk in answer.used_chunks] == [1, 2, 3]


def test_threshold_abstain_still_reports_what_was_retrieved():
    # Diagnosis matters: "the right section was never retrieved" and "it was
    # retrieved but the threshold rejected it" need different fixes.
    engine = build_engine([candidate("weak", 0)], [0.05], "should never be used")

    answer = engine.answer("unrelated question")

    assert answer.abstained
    assert [chunk.chunk.text for chunk in answer.used_chunks] == ["weak"]


def test_abstain_with_nothing_retrieved_reports_an_empty_context():
    engine = build_engine([], [], "should never be used")

    answer = engine.answer("unrelated question")

    assert answer.abstained
    assert answer.used_chunks == []


def test_embedder_and_reranker_are_freed_before_calling_the_llm():
    engine = build_engine([candidate("relevant", 0)], [0.9], "Grounded answer [1].")

    engine.answer("question")

    assert engine.embedder.unloaded
    assert engine.reranker.unloaded


def test_gpu_is_not_freed_on_abstain_since_the_llm_is_never_called():
    engine = build_engine([candidate("weak", 0)], [0.05], "should never be used")

    engine.answer("unrelated question")

    assert not engine.embedder.unloaded
    assert not engine.reranker.unloaded


def test_stream_sends_citations_first_then_tokens_then_done():
    engine = build_engine([candidate("relevant", 0)], [0.9], "Grounded answer [1].")

    events = list(engine.stream_answer("question"))

    assert events[0].event == "citations"
    assert [event.event for event in events[1:-1]] == ["token"] * (len(events) - 2)
    assert events[-1].event == "done"
    assert "".join(event.token for event in events[1:-1]) == "Grounded answer [1]."
    assert events[-1].answer.citations[0].marker == 1


def test_stream_never_leaks_the_abstain_sentinel_as_tokens():
    engine = build_engine([candidate("relevant", 0)], [0.9], ABSTAIN_SENTINEL)

    events = list(engine.stream_answer("question"))

    assert [event.event for event in events] == ["citations", "done"]
    assert events[-1].answer.abstained


class TestVerification:
    def test_answer_carries_verified_claims_when_enabled(self):
        engine = build_engine([candidate("relevant", 0)], [0.9], "DPR is the retriever [1].")

        answer = engine.answer("question")

        assert [claim.verdict for claim in answer.verified_claims] == [ClaimVerdict.SUPPORTED]
        assert engine.verifier.calls == 1

    def test_verification_is_skipped_when_disabled(self):
        engine = build_engine(
            [candidate("relevant", 0)],
            [0.9],
            "DPR is the retriever [1].",
            verification_enabled=False,
        )

        answer = engine.answer("question")

        assert answer.verified_claims == []
        assert engine.verifier.calls == 0

    def test_abstained_answers_are_never_verified(self):
        # Nothing was claimed, so there is nothing to check.
        engine = build_engine([candidate("weak", 0)], [0.05], "should never be used")

        answer = engine.answer("question")

        assert answer.abstained
        assert answer.verified_claims == []
        assert engine.verifier.calls == 0

    def test_a_sentinel_abstain_is_not_verified_either(self):
        engine = build_engine([candidate("relevant", 0)], [0.9], ABSTAIN_SENTINEL)

        answer = engine.answer("question")

        assert answer.abstained
        assert answer.verified_claims == []
        assert engine.verifier.calls == 0

    def test_streaming_reports_verification_in_the_final_done_event(self):
        engine = build_engine([candidate("relevant", 0)], [0.9], "DPR is the retriever [1].")

        events = list(engine.stream_answer("question"))

        assert events[-1].event == "done"
        verdicts = [claim.verdict for claim in events[-1].answer.verified_claims]
        assert verdicts == [ClaimVerdict.SUPPORTED]


class TestPreviewRetrieval:
    """Retrieval-only path: for ablation runs that only need recall@k/MRR, not
    an actual answer. Must never touch the LLM - that is the whole point."""

    def test_returns_chunks_and_sufficient_true_above_threshold(self):
        engine = build_engine([candidate("relevant", 0)], [0.9], "should never be used")

        chunks, sufficient = engine.preview_retrieval("question")

        assert sufficient
        assert [chunk.chunk.text for chunk in chunks] == ["relevant"]
        assert engine.llm.calls == 0

    def test_returns_chunks_and_sufficient_false_below_threshold(self):
        engine = build_engine([candidate("weak", 0)], [0.05], "should never be used")

        chunks, sufficient = engine.preview_retrieval("question")

        assert not sufficient
        assert [chunk.chunk.text for chunk in chunks] == ["weak"]  # still diagnosable
        assert engine.llm.calls == 0

    def test_returns_empty_and_sufficient_false_when_nothing_retrieved(self):
        engine = build_engine([], [], "should never be used")

        chunks, sufficient = engine.preview_retrieval("question")

        assert not sufficient
        assert chunks == []
        assert engine.llm.calls == 0

    def test_never_unloads_the_gpu_since_no_llm_call_follows(self):
        # unload-before-LLM only makes sense when an LLM call is about to
        # happen; here it would just force a ~25s reload for nothing.
        engine = build_engine([candidate("relevant", 0)], [0.9], "unused")

        engine.preview_retrieval("question")

        assert not engine.embedder.unloaded
        assert not engine.reranker.unloaded


class TestSummarize:
    def test_a_summary_cites_sections_and_leaves_used_chunks_empty(self):
        # No retrieval ran, so there are no scored chunks to show - the section
        # summaries are what a reader inspects instead.
        chunks = [make_chunk("real source text", index=0, section_path=["2 Methods"])]
        summaries = [
            SectionSummary(
                marker=1, section="2 Methods", text="summary", chunk_ids=[chunks[0].chunk_id]
            )
        ]
        engine = build_engine(
            [],
            [],
            "unused",
            store=FakeStore(chunks),
            summarizer=FakeSummarizer("Final answer [1].", summaries),
        )

        answer = engine.summarize(None, "doc-1")

        assert answer.used_chunks == []
        assert [summary.section for summary in answer.section_summaries] == ["2 Methods"]
        assert [citation.marker for citation in answer.citations] == [1]

    def test_an_unknown_document_abstains_instead_of_summarising_nothing(self):
        engine = build_engine(
            [], [], "unused", store=FakeStore([]), summarizer=FakeSummarizer("", [])
        )

        answer = engine.summarize(None, "missing")

        assert answer.text == ABSTAIN_MESSAGE
        assert answer.section_summaries == []
        assert engine.summarizer.calls == []  # never reached: nothing to summarise

    def test_summary_claims_are_verified_against_section_source_text_not_the_summary(self):
        # The judge must receive the real chunk text, never the intermediate
        # summary - otherwise a hallucinated section summary validates itself.
        chunks = [make_chunk("real source text", index=0, section_path=["2 Methods"])]
        summaries = [
            SectionSummary(
                marker=1,
                section="2 Methods",
                text="invented summary",
                chunk_ids=[chunks[0].chunk_id],
            )
        ]
        engine = build_engine(
            [],
            [],
            "unused",
            store=FakeStore(chunks),
            summarizer=FakeSummarizer("Final answer [1].", summaries),
        )

        engine.summarize(None, "doc-1")

        assert engine.verifier.last_passages == {1: "real source text"}

    def test_verification_is_skipped_for_summaries_when_disabled(self):
        chunks = [make_chunk("real source text", index=0, section_path=["2 Methods"])]
        summaries = [
            SectionSummary(marker=1, section="2 Methods", text="s", chunk_ids=[chunks[0].chunk_id])
        ]
        engine = build_engine(
            [],
            [],
            "unused",
            verification_enabled=False,
            store=FakeStore(chunks),
            summarizer=FakeSummarizer("Final answer [1].", summaries),
        )

        answer = engine.summarize(None, "doc-1")

        assert answer.verified_claims == []
        assert engine.verifier.calls == 0

    def test_no_question_falls_back_to_a_plain_document_summary_request(self):
        chunks = [make_chunk("text", index=0, section_path=["1 Intro"])]
        engine = build_engine(
            [], [], "unused", store=FakeStore(chunks), summarizer=FakeSummarizer("Final.", [])
        )

        engine.summarize(None, "doc-1")

        question_asked, _ = engine.summarizer.calls[0]
        assert question_asked  # not None/empty - the summarizer always gets a request


class TestSummaryFallback:
    """A global question the keyword router misses ("what is this research
    about" for a Wikipedia article - "research" wasn't a recognised document
    word) should still get answered: when retrieval gives up, answer()/
    stream_answer() retry via summarize() before finalising an abstain."""

    def test_a_weak_retrieval_falls_back_to_a_summary_when_one_succeeds(self):
        chunks = [make_chunk("real source text", index=0, section_path=["2 Methods"])]
        summaries = [
            SectionSummary(marker=1, section="2 Methods", text="s", chunk_ids=[chunks[0].chunk_id])
        ]
        engine = build_engine(
            [candidate("weak", 0)],
            [0.05],  # below score_threshold - selection is insufficient
            "unused",
            store=FakeStore(chunks),
            summarizer=FakeSummarizer("The paper studies X [1].", summaries),
        )

        answer = engine.answer("what is this research about", "doc-1")

        assert not answer.abstained
        assert answer.text == "The paper studies X [1]."

    def test_still_abstains_when_the_document_has_nothing_to_summarise_either(self):
        engine = build_engine(
            [candidate("weak", 0)],
            [0.05],
            "unused",
            store=FakeStore([]),
            summarizer=FakeSummarizer("", []),
        )

        answer = engine.answer("what is this research about", "doc-1")

        assert answer.abstained

    def test_all_documents_mode_never_falls_back_to_a_summary(self):
        # summarize() needs one document to scope to (rag.md 8) - a
        # cross-paper conversation (document_id=None) has no single document
        # to fall back to, so it must stay a plain abstain.
        engine = build_engine(
            [candidate("weak", 0)],
            [0.05],
            "unused",
            store=FakeStore([make_chunk("text", index=0)]),
            summarizer=FakeSummarizer("would never be reached", []),
        )

        answer = engine.answer("what is this research about", document_id=None)

        assert answer.abstained
        assert engine.summarizer.calls == []

    def test_stream_answer_falls_back_to_a_summary_with_the_normal_event_shape(self):
        chunks = [make_chunk("real source text", index=0, section_path=["2 Methods"])]
        summaries = [
            SectionSummary(marker=1, section="2 Methods", text="s", chunk_ids=[chunks[0].chunk_id])
        ]
        engine = build_engine(
            [candidate("weak", 0)],
            [0.05],
            "unused",
            store=FakeStore(chunks),
            summarizer=FakeSummarizer("The paper studies X [1].", summaries),
        )

        events = list(engine.stream_answer("what is this research about", "doc-1"))

        assert [event.event for event in events] == ["citations", "token", "done"]
        assert events[1].token == "The paper studies X [1]."
        assert events[2].answer.text == "The paper studies X [1]."

    def test_a_mid_stream_sentinel_falls_back_without_a_second_citations_event(self):
        # Retrieval succeeds (above threshold) but the model itself signals
        # abstain - the fallback still applies here since nothing has
        # reached the client yet (still held), but citations for this turn
        # were already sent once, right after retrieval.
        chunks = [make_chunk("real source text", index=0, section_path=["2 Methods"])]
        summaries = [
            SectionSummary(marker=1, section="2 Methods", text="s", chunk_ids=[chunks[0].chunk_id])
        ]
        engine = build_engine(
            [candidate("relevant", 0)],
            [0.9],
            ABSTAIN_SENTINEL,
            store=FakeStore(chunks),
            summarizer=FakeSummarizer("The paper studies X [1].", summaries),
        )

        events = list(engine.stream_answer("what is this research about", "doc-1"))

        assert [event.event for event in events] == ["citations", "token", "done"]
        assert events[1].token == "The paper studies X [1]."
        assert not events[2].answer.abstained


class TestAsk:
    def test_a_specific_question_is_answered_normally(self):
        engine = build_engine([candidate("relevant", 0)], [0.9], "Grounded answer [1].")

        answer = engine.ask("What retriever does this paper use?", "doc-1")

        assert answer.section_summaries == []
        assert answer.text == "Grounded answer [1]."

    def test_a_global_question_with_a_document_id_routes_to_summarize(self):
        chunks = [make_chunk("real source text", index=0, section_path=["2 Methods"])]
        summaries = [
            SectionSummary(marker=1, section="2 Methods", text="s", chunk_ids=[chunks[0].chunk_id])
        ]
        engine = build_engine(
            [],
            [],
            "unused",
            store=FakeStore(chunks),
            summarizer=FakeSummarizer("Final [1].", summaries),
        )

        answer = engine.ask("Summarize this paper.", "doc-1")

        assert answer.section_summaries != []
        assert engine.summarizer.calls

    def test_a_global_question_without_a_document_id_falls_back_to_answering(self):
        # summarize() needs one document to scope to - there is no "summarize
        # everything ever ingested" design, so an unscoped global-sounding
        # question still goes through retrieval like any other question.
        engine = build_engine(
            [candidate("relevant", 0)],
            [0.9],
            "Grounded answer [1].",
            summarizer=FakeSummarizer("unused", []),
        )

        answer = engine.ask("Summarize this paper.")

        assert answer.section_summaries == []
        assert engine.summarizer.calls == []


class TestBuildTree:
    def test_builds_from_the_leaves_and_replaces_the_old_tree(self):
        leaves = [make_chunk("a", index=0), make_chunk("b", index=1)]
        node = make_chunk("summary", index=0)
        store = FakeStore(leaves)
        engine = build_engine([], [], "unused", store=store, tree_builder=FakeTreeBuilder([node]))

        count = engine.build_tree("doc-1")

        assert count == 1
        assert store.deleted_trees == ["doc-1"]
        assert store.upserted == [node]
        assert engine.tree_builder.built_from == [leaves]

    def test_a_document_with_no_chunks_builds_nothing(self):
        store = FakeStore([])
        engine = build_engine([], [], "unused", store=store, tree_builder=FakeTreeBuilder([]))

        assert engine.build_tree("missing") == 0
        assert store.deleted_trees == []


def tree_node(text: str, section: str, leaves: list, position: int = 0):
    """A level-1 node the way TreeBuilder makes one: its own ID, pointing at leaves."""
    base = make_chunk(text, index=position, section=section)
    return base.model_copy(
        update={
            "chunk_id": node_id_for(base.document_id, 1, position),
            "metadata": base.metadata.model_copy(
                update={"level": 1, "source_ids": [leaf.chunk_id for leaf in leaves]}
            ),
        }
    )


class TestTreeSummaries:
    def test_raptor_summaries_reduce_the_level_one_nodes_without_a_map_step(self):
        leaves = [make_chunk("leaf text", index=0, section="2 Methods")]
        node = tree_node("methods summary", "2 Methods", leaves)
        summarizer = FakeSummarizer("Final [1].", [])
        engine = build_engine(
            [],
            [],
            "unused",
            store=FakeStore(leaves, nodes={1: [node]}),
            summarizer=summarizer,
            summary_method="raptor",
        )

        answer = engine.summarize(None, "doc-1")

        assert summarizer.calls == []  # no per-section map calls at question time
        _, summaries = summarizer.reduced
        assert [(s.marker, s.section, s.text) for s in summaries] == [
            (1, "2 Methods", "methods summary")
        ]
        assert [c.marker for c in answer.citations] == [1]

    def test_tree_summary_claims_are_verified_against_the_leaves(self):
        leaves = [make_chunk("the real leaf text", index=0, section="2 Methods")]
        node = tree_node("invented summary", "2 Methods", leaves)
        engine = build_engine(
            [],
            [],
            "unused",
            store=FakeStore(leaves, nodes={1: [node]}),
            summarizer=FakeSummarizer("Final [1].", []),
            summary_method="raptor",
        )

        engine.summarize(None, "doc-1")

        assert engine.verifier.last_passages == {1: "the real leaf text"}

    def test_a_document_without_a_tree_falls_back_to_map_reduce(self):
        leaves = [make_chunk("leaf", index=0)]
        summarizer = FakeSummarizer("Final.", [])
        engine = build_engine(
            [],
            [],
            "unused",
            store=FakeStore(leaves),
            summarizer=summarizer,
            summary_method="raptor",
        )

        engine.summarize(None, "doc-1")

        assert len(summarizer.calls) == 1


class TestTreeNodeCitations:
    def test_a_cited_tree_node_is_verified_against_its_leaves(self):
        leaf = make_chunk("the real leaf text", index=0)
        node = tree_node("invented summary", "2 Methods", [leaf], position=7)
        store = FakeStore([leaf])
        engine = build_engine(
            [RetrievedChunk(chunk=node, score=0.9)], [0.9], "Answer [1].", store=store
        )

        engine.answer("q")

        assert engine.verifier.last_passages == {1: "the real leaf text"}
        assert store.lookups == [[leaf.chunk_id]]

    def test_answers_from_leaves_only_keep_the_original_verification_path(self):
        engine = build_engine([candidate("leaf", 0)], [0.9], "Answer [1].", store=FakeStore([]))

        engine.answer("q")

        assert not hasattr(engine.verifier, "last_passages")


class TestFigureVision:
    def test_a_figure_chunk_is_described_with_the_actual_question(self, tmp_path):
        image_path = str(tmp_path / "figure.png")
        vision = FakeVisionDescriber("a bar chart showing accuracy rising with more documents")
        engine = build_engine(
            [candidate("generic ingest-time caption", index=0, image_paths=[image_path])],
            [0.9],
            "The chart shows [1].",
            vision=vision,
        )

        engine.answer("What does the chart show?")

        assert len(vision.calls) == 1
        called_path, called_question = vision.calls[0]
        assert str(called_path) == image_path
        assert called_question == "What does the chart show?"
        context_text = engine.llm.last_messages[-1]["content"]
        assert "a bar chart showing accuracy rising" in context_text
        assert "generic ingest-time caption" not in context_text

    def test_a_chunk_without_images_is_never_sent_to_vision(self):
        vision = FakeVisionDescriber("should never be used")
        engine = build_engine(
            [candidate("plain text", index=0)], [0.9], "Answer [1].", vision=vision
        )

        engine.answer("a normal question")

        assert vision.calls == []

    def test_vision_is_a_no_op_when_not_configured(self):
        engine = build_engine(
            [candidate("generic caption", index=0, image_paths=["some/path.png"])],
            [0.9],
            "Answer [1].",
        )  # vision defaults to None

        answer = engine.answer("a question")

        assert answer.used_chunks[0].chunk.text == "generic caption"

    def test_a_failed_vision_call_falls_back_to_the_original_text(self):
        vision = FakeVisionDescriber("")  # simulates describe() returning "" on failure
        engine = build_engine(
            [candidate("generic caption", index=0, image_paths=["some/path.png"])],
            [0.9],
            "Answer [1].",
            vision=vision,
        )

        answer = engine.answer("a question")

        assert answer.used_chunks[0].chunk.text == "generic caption"
