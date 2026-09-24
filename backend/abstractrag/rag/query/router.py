"""Deciding whether a question needs the whole document, not just the top-k.

A specific question ("what retriever does this paper use?") is answered by
retrieval; a global one ("summarize this paper") cannot be - the answer is
spread across every section, so top-k structurally misses most of it (rag.md
section 8). This is a plain keyword match rather than an LLM call: the system
is English-only end to end (papers and Wikipedia, both English - rag.md 0/2.4),
so a fixed set of English trigger phrases covers this cheaply, without adding
a round-trip to every question the way an LLM classifier would (rag.md 7.4.1).
"""

import re

# Multi-word phrases, not bare words: "main contribution" is a global signal,
# but "main" alone would also match "what is the main dataset", which is not.
# The same principle applies to "summary"/"overview"/"key results": a question
# is anchored to "the document" when it names it directly - this is what keeps
# "summarize this paper" global while "summary statistic" (Table 3) stays
# specific. "high-level overview", "tl;dr" etc. need no anchor: they are
# inherently about the whole thing being discussed, not one detail of it.
_DOC = r"(?:this|the) (?:paper|document|article|study)"
_SUMMARIZE = r"summar(?:y|ies|ize[sd]?|ise[sd]?|izing|ising)"

_GLOBAL_PATTERNS = re.compile(
    "|".join(
        [
            rf"\b{_SUMMARIZE}\b.{{0,30}}\b{_DOC}\b",
            rf"\b{_DOC}\b.{{0,30}}\b{_SUMMARIZE}\b",
            r"\btl;?\s?dr\b",
            r"\bin a nutshell\b",
            r"\bgist of\b",
            r"\bhigh-level (?:summary|overview)\b",
            rf"\boverview of {_DOC}\b",
            r"\bmain (?:contribution|idea|point|takeaway|argument)s?\b",
            r"\bkey (?:finding|takeaway|point)s?\b",
            rf"\bwhat (?:is|are) {_DOC} about\b",
            rf"\bwhat does {_DOC} (?:do|cover|discuss)\b",
        ]
    ),
    re.IGNORECASE,
)


def is_global_question(question: str) -> bool:
    """True when a question needs the whole document rather than top-k chunks."""
    return bool(_GLOBAL_PATTERNS.search(question))
