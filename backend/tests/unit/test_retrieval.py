from abstractrag.rag.retrieval import rrf
from tests.conftest import make_chunk


def test_chunk_found_by_both_retrievers_wins():
    shared = make_chunk("hybrid hit", index=0)
    dense_only = make_chunk("dense hit", index=1)
    sparse_only = make_chunk("sparse hit", index=2)

    fused = rrf.fuse(
        [
            [(dense_only, 0.91), (shared, 0.80)],
            [(sparse_only, 12.0), (shared, 9.0)],
        ],
        k=60,
    )

    assert fused[0].chunk.chunk_id == shared.chunk_id
    assert len(fused) == 3


def test_fusion_ignores_incomparable_score_scales():
    chunk_a = make_chunk("a", index=0)
    chunk_b = make_chunk("b", index=1)

    # chunk_b has a far larger raw score but a worse rank in both lists.
    fused = rrf.fuse([[(chunk_a, 0.01), (chunk_b, 999.0)]], k=60)

    assert fused[0].chunk.chunk_id == chunk_a.chunk_id


def test_higher_k_flattens_the_ranking():
    chunk_a = make_chunk("a", index=0)
    chunk_b = make_chunk("b", index=1)
    results = [[(chunk_a, 1.0), (chunk_b, 0.9)]]

    small_k = rrf.fuse(results, k=1)
    large_k = rrf.fuse(results, k=1000)

    assert small_k[0].score - small_k[1].score > large_k[0].score - large_k[1].score
