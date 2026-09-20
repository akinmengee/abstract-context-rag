"""Engine behaviour with fake components: abstain rules, citations, streaming."""

from collections.abc import Iterator

from abstractrag.core.config import RetrievalSettings, Settings
from abstractrag.rag.engine import RagEngine
from abstractrag.rag.generation.prompts import ABSTAIN_MESSAGE, ABSTAIN_SENTINEL
from abstractrag.rag.models import RetrievedChunk
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

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        return self.response

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        self.calls += 1
        for index in range(0, len(self.response), 4):
            yield self.response[index : index + 4]


def build_engine(
    candidates: list[RetrievedChunk], scores: list[float], response: str
) -> RagEngine:
    settings = Settings(retrieval=RetrievalSettings(context_size=3, score_threshold=0.3))
    return RagEngine(
        settings=settings,
        resolver=None,
        chunker=None,
        embedder=FakeEmbedder(),
        store=None,
        retriever=FakeRetriever(candidates),
        reranker=FakeReranker(scores),
        llm=FakeLLM(response),
    )


def candidate(text: str, index: int, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(text, index=index), score=score)


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
