"""Retrieval metrics, computed over the ranked section names a question retrieved.

Ground truth is a section name ("2.2 Retriever: DPR"), not a chunk ID: chunk IDs
shift whenever chunking parameters change, and comparing chunking variants is
exactly what the ablation runs do. Section names survive that.

recall@k and MRR both derive from one primitive - where the expected section
first appears in the ranking - so there is only one matching rule to get right.
"""


def _normalize(section: str) -> str:
    return section.strip().casefold()


def first_hit_rank(retrieved_sections: list[str], expected_section: str) -> int | None:
    """1-based rank of the expected section, or None if it was never retrieved."""
    expected = _normalize(expected_section)
    for rank, section in enumerate(retrieved_sections, start=1):
        if _normalize(section) == expected:
            return rank
    return None


def recall_at_k(retrieved_sections: list[str], expected_section: str, k: int) -> bool:
    """Did the expected section make it into the top k?"""
    rank = first_hit_rank(retrieved_sections, expected_section)
    return rank is not None and rank <= k


def reciprocal_rank(retrieved_sections: list[str], expected_section: str) -> float:
    """1/rank of the expected section, 0.0 when it was never retrieved.

    Averaged over questions this is MRR: it rewards ranking the right section
    high, not merely including it somewhere in the list.
    """
    rank = first_hit_rank(retrieved_sections, expected_section)
    return 1.0 / rank if rank else 0.0
