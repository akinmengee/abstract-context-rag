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
# The same principle applies to "summary"/"overview": a question is anchored
# to "the document" when it names it directly - this is what keeps "summarize
# this paper" global while "summary statistic" (Table 3) stays specific.
# "high-level overview", "tl;dr" etc. need no anchor: they are inherently
# about the whole thing being discussed, not one detail of it.
#
# Covers both source types this project actually ingests (papers, Wikipedia
# articles) plus generic ways of referring to either - "research" and
# "source" in particular were measured missing live: "what is this research
# about" fell through to plain retrieval instead of summarize() for a
# Wikipedia article, which isn't a paper and so never said "paper"/"document".
_DOC = r"(?:this|the) (?:paper|document|article|study|research|source|text|content)"
_SUMMARIZE = r"summar(?:y|ies|ize[sd]?|ise[sd]?|izing|ising)"

# The gap between a trigger word and the document reference must not cross a
# sentence boundary - otherwise "I read the summary of related work. Does
# this paper also cover TriviaQA?" pairs words from two unrelated sentences.
_GAP = r"[^.!?]{0,30}"

_GLOBAL_PATTERNS = re.compile(
    "|".join(
        [
            rf"\b{_SUMMARIZE}\b{_GAP}\b{_DOC}\b",
            rf"\b{_DOC}\b{_GAP}\b{_SUMMARIZE}\b",
            r"\btl;?\s?dr\b",
            r"\bin a nutshell\b",
            r"\bgist of\b",
            r"\bhigh-level (?:summary|overview)\b",
            rf"\boverview of {_DOC}\b",
            rf"\bwhat (?:is|are) {_DOC} about\b",
            rf"\bwhat does {_DOC} (?:do|cover|discuss)\b",
        ]
    ),
    re.IGNORECASE,
)

# "main contribution"/"key finding" etc. are global without ever naming the
# document ("what are the key findings?"), so - unlike every pattern above -
# these are not anchored to _DOC. That leaves them able to fire on a question
# about one exhibit that happens to use the same words ("what key finding
# does Table 3 report?"), so a nearby "Table N"/"Figure N" reference vetoes
# the match below instead.
_NARROW_GLOBAL_PATTERNS = re.compile(
    r"\bmain (?:contribution|idea|point|takeaway|argument|finding|result)s?\b"
    r"|\bkey (?:finding|takeaway|point|result)s?\b",
    re.IGNORECASE,
)

_SPECIFIC_REFERENCE = re.compile(r"\b(?:table|figure)\s*\d", re.IGNORECASE)


def is_global_question(question: str) -> bool:
    """True when a question needs the whole document rather than top-k chunks."""
    if _GLOBAL_PATTERNS.search(question):
        return True
    if _NARROW_GLOBAL_PATTERNS.search(question):
        return not _SPECIFIC_REFERENCE.search(question)
    return False
