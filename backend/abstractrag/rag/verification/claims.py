"""Splitting an answer into the individual claims a verifier can check.

One sentence, one claim. Answers here are short and already carry their sources
inline ("DPR is the retriever [1][3]."), so a sentence maps cleanly onto "one
thing asserted, and what it leans on" - no LLM call needed to find the seams.

A sentence carrying no marker is still a claim: an assertion with no source is
exactly the kind of thing verification exists to surface.
"""

import re
from dataclasses import dataclass, field

from abstractrag.rag.generation.prompts import CITATION_MARKER

# Split on sentence punctuation only when whitespace follows, which leaves
# decimals ("0.88") and section numbers ("4.3") intact.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Claim:
    text: str
    markers: list[int] = field(default_factory=list)


def split_claims(answer: str) -> list[Claim]:
    claims = []
    for part in _SENTENCE_BOUNDARY.split(answer):
        sentence = part.strip()
        if sentence:
            claims.append(Claim(text=sentence, markers=_markers_in(sentence)))
    return claims


def _markers_in(sentence: str) -> list[int]:
    """Marker numbers in first-appearance order, without repeats."""
    seen: dict[int, None] = {}
    for marker in CITATION_MARKER.findall(sentence):
        seen.setdefault(int(marker), None)
    return list(seen)
