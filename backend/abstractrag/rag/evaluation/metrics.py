"""Retrieval metrics, computed over where the retrieved chunks came from.

Ground truth is a (paper, section name) pair, not a chunk ID: chunk IDs shift
whenever chunking parameters change, and comparing chunking variants is exactly
what the ablation runs do. Section names survive that; the paper is part of the
key because every paper has a "1 Introduction".

recall@k and MRR both derive from one primitive - where the expected section
first appears in the ranking - so there is only one matching rule to get right.
"""

from abstractrag.rag.evaluation.models import Evidence, RetrievedSection


def _normalize(section: str) -> str:
    return section.strip().casefold()


def matches(retrieved: RetrievedSection, evidence: Evidence) -> bool:
    return evidence.paper in retrieved.origin and _normalize(retrieved.section) == _normalize(
        evidence.section
    )


def first_hit_rank(retrieved: list[RetrievedSection], evidence: Evidence) -> int | None:
    """1-based rank of the expected section, or None if it was never retrieved."""
    for rank, item in enumerate(retrieved, start=1):
        if matches(item, evidence):
            return rank
    return None


def recall_at_k(retrieved: list[RetrievedSection], evidence: Evidence, k: int) -> bool:
    """Did the expected section make it into the top k?"""
    rank = first_hit_rank(retrieved, evidence)
    return rank is not None and rank <= k


def reciprocal_rank(retrieved: list[RetrievedSection], evidence: Evidence) -> float:
    """1/rank of the expected section, 0.0 when it was never retrieved.

    Averaged over questions this is MRR: it rewards ranking the right section
    high, not merely including it somewhere in the list.
    """
    rank = first_hit_rank(retrieved, evidence)
    return 1.0 / rank if rank else 0.0


def evidence_recall(retrieved: list[RetrievedSection], evidence: list[Evidence]) -> float:
    """Share of the expected evidence found anywhere in what the LLM saw.

    Partial credit on purpose: a multi-hop question scoring 0.5 found the first
    hop and missed the second, which is a different failure from finding nothing.
    """
    if not evidence:
        return 0.0
    found = sum(first_hit_rank(retrieved, item) is not None for item in evidence)
    return found / len(evidence)
