from abstractrag.rag.generation.prompts import build_context, order_for_context
from abstractrag.rag.models import RetrievedChunk
from tests.conftest import make_chunk


def retrieved(text: str, score: float, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(text, index=index), score=score)


def test_context_blocks_are_numbered_and_cited():
    context, citations = build_context([retrieved("first", 0.9, 0), retrieved("second", 0.5, 1)])

    assert context.startswith("[1] Test Paper — Results — page 1\nfirst")
    assert [citation.marker for citation in citations] == [1, 2]
    assert citations[0].title == "Test Paper"
    assert citations[1].page == 2


def test_best_chunks_are_placed_at_the_edges():
    chunks = [retrieved(f"c{i}", score, i) for i, score in enumerate([0.1, 0.9, 0.5, 0.7, 0.3])]

    ordered = order_for_context(chunks)

    assert ordered[0].score == 0.9
    assert ordered[-1].score == 0.7
    assert len(ordered) == len(chunks)


def test_ordering_uses_rerank_score_when_present():
    weak_but_reranked = retrieved("a", 0.1, 0)
    weak_but_reranked.rerank_score = 0.99
    strong_first_stage = retrieved("b", 0.8, 1)

    ordered = order_for_context([strong_first_stage, weak_but_reranked])

    assert ordered[0].chunk.text == "a"
