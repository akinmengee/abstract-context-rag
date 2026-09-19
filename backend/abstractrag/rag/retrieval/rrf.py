"""Reciprocal Rank Fusion.

score(chunk) = sum over result lists of 1 / (k + rank)

Fusing ranks instead of scores is what makes dense and sparse results combinable:
cosine similarity and BM25-style lexical scores are not on the same scale, but a
rank is a rank. k (default 60) damps the influence of the top positions.
"""

from collections import defaultdict
from collections.abc import Sequence

from abstractrag.rag.models import Chunk, RetrievedChunk


def fuse(
    result_lists: Sequence[Sequence[tuple[Chunk, float]]], k: int = 60
) -> list[RetrievedChunk]:
    fused_scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}

    for results in result_lists:
        for rank, (chunk, _score) in enumerate(results, start=1):
            fused_scores[chunk.chunk_id] += 1.0 / (k + rank)
            chunks[chunk.chunk_id] = chunk

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [RetrievedChunk(chunk=chunks[chunk_id], score=score) for chunk_id, score in ranked]
