"""Prompts and reply parsers for the agent's three decisions.

Each reply is kept to a tiny fixed format - numbers, one query line, or DONE -
because a 4B model follows a one-line format far more reliably than prose, and
a parser that can say "unreadable" is what lets the caller fall back safely.
"""

import re

from abstractrag.rag.models import RetrievedChunk

# Enough of a chunk to judge it; the full text only costs context budget.
_PASSAGE_CHARS = 1500
_NOTE_CHARS = 500

GRADE_SYSTEM = """You decide which passages help answer a question.

Reply with the numbers of the passages that contain information needed to \
answer it, comma-separated, like: 1, 3
Reply NONE if no passage does.

A passage counts if it states a fact the answer needs - including a fact about \
something the question depends on indirectly, such as the component or earlier \
method the question's subject is built on. A passage about an unrelated system \
that merely shares a metric, dataset or word with the question does not count."""

REWRITE_SYSTEM = """You turn a question into a search query for a collection of \
research papers. The first search for it found nothing useful.

Reply with only the new search query: the key names and terms, no question words, \
no explanation."""

_SEARCH_ADVICE = """Check every link the question relies on. A question often names \
something indirectly ("RAG's retriever", "the model this paper builds on"); the \
passages only answer it if they also say what that thing is. If none of them \
does, search for that, e.g. SEARCH: which retriever does RAG use
When a passage names the method, model or paper the question depends on (for \
example "our retriever is based on DPR"), search for that name directly.
Each side of a comparison needs its own passages."""

PLAN_SYSTEM = f"""You plan searches over a collection of research papers to answer \
a question. You see the question and the passages found so far.

If the passages contain everything needed to answer the question, reply with \
exactly: DONE
If a piece is missing, reply with exactly one line: SEARCH: <search query for \
the missing piece>

{_SEARCH_ADVICE}"""

# The first follow-up has no DONE option: a small model declares a first search
# "enough" even when a comparison has only one side or a link is unstated
# (measured on qwen3 4B), and one extra search costs seconds, not a wrong answer.
FOLLOW_UP_SYSTEM = f"""You plan searches over a collection of research papers to \
answer a question. You see the question and the passages found so far.

Reply with exactly one line: SEARCH: <search query for the piece of information \
most likely still missing, or for a link the question relies on that the \
passages do not state>

{_SEARCH_ADVICE}"""


def _label(chunk: RetrievedChunk) -> str:
    metadata = chunk.chunk.metadata
    return f"{metadata.title} - {metadata.section or 'untitled'}"


def build_grade_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    passages = "\n\n".join(
        f"[{number}] ({_label(chunk)})\n{chunk.chunk.text[:_PASSAGE_CHARS]}"
        for number, chunk in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": GRADE_SYSTEM},
        {
            "role": "user",
            "content": f"Question: {question}\n\nPassages:\n{passages}\n\n"
            "Which passages help answer the question? Reply with numbers or NONE.",
        },
    ]


def build_rewrite_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"Question: {question}"},
    ]


def build_plan_messages(
    question: str, chunks: list[RetrievedChunk], searches: list[str], allow_done: bool = True
) -> list[dict[str, str]]:
    notes = "\n\n".join(
        f"- ({_label(chunk)}) {chunk.chunk.text[:_NOTE_CHARS]}" for chunk in chunks
    ) or "(none yet)"
    ask = "Reply DONE or SEARCH: <query>." if allow_done else "Reply SEARCH: <query>."
    return [
        {"role": "system", "content": PLAN_SYSTEM if allow_done else FOLLOW_UP_SYSTEM},
        {
            "role": "user",
            "content": f"Question: {question}\n\nSearches so far: {'; '.join(searches)}\n\n"
            f"Passages found so far:\n{notes}\n\n{ask}",
        },
    ]


def parse_grade(reply: str, count: int) -> set[int] | None:
    """Passage numbers the grader kept; empty for NONE, None if unreadable."""
    numbers = {int(n) for n in re.findall(r"\d+", reply) if 1 <= int(n) <= count}
    if numbers:
        return numbers
    if re.search(r"\bnone\b", reply, re.IGNORECASE):
        return set()
    return None


def parse_rewrite(reply: str) -> str:
    """First non-empty line, without quotes or a "Query:" label; "" if none."""
    for line in reply.splitlines():
        line = re.sub(r"^\s*(search\s+)?query\s*:\s*", "", line, flags=re.IGNORECASE)
        line = line.strip().strip("\"'`").strip()
        if line:
            return line
    return ""


def parse_plan(reply: str) -> str | None:
    """The next search query, or None for DONE - and for anything unreadable,
    because stopping early still answers from what was found."""
    match = re.search(r"^\s*SEARCH\s*:\s*(.+)$", reply, re.IGNORECASE | re.MULTILINE)
    if match:
        # Measured: the model sometimes packs two searches into one line
        # ("... DPR use SEARCH: ... ColBERT use") - take the first one.
        first = re.split(r"\bSEARCH\s*:", match.group(1), flags=re.IGNORECASE)[0]
        query = first.strip().strip("\"'`").strip()
        return query or None
    return None
