"""Prompts for the two summarisation stages.

They differ because their inputs differ: the map stage sees real source text and
must stay extractive, while the reduce stage only ever sees section summaries and
must carry their numbers through - those numbers are the summary's citations.
"""

from abstractrag.rag.models import SectionSummary
from abstractrag.rag.summarization.sections import SectionGroup

# What a summary is asked for when the caller had no question of their own.
DEFAULT_REQUEST = "Summarise this document."

# A section that says nothing about the request is dropped rather than summarised
# into filler the reduce stage would then have to ignore.
NOTHING_RELEVANT = "NOTHING_RELEVANT"

MAP_SYSTEM = f"""You summarise one section of a document.

Rules:
- Use only the section text. Never add outside knowledge.
- Quote numbers, dataset names and results exactly as written.
- Write at most 4 sentences. No preamble, no heading, no bullet points.
- If the section says nothing about the request, reply with exactly \
{NOTHING_RELEVANT} and nothing else."""

MAP_USER = """Section: {section}

{text}

Summarise this section with respect to: {question}"""

REDUCE_SYSTEM = """You answer a request using numbered section summaries of one document.

Rules:
- Use only the summaries. Never add outside knowledge.
- Cite the section number in square brackets after every sentence, like [2]. One \
sentence can rest on several sections: [1][3].
- Quote numbers and names exactly as written.
- Be concise. Do not repeat the request and do not explain your reasoning."""

REDUCE_USER = """Section summaries:
{summaries}

Request: {question}"""


def build_map_messages(question: str, group: SectionGroup) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MAP_SYSTEM},
        {
            "role": "user",
            "content": MAP_USER.format(
                section=group.section, text=group.text, question=question
            ),
        },
    ]


def build_reduce_messages(
    question: str, summaries: list[SectionSummary]
) -> list[dict[str, str]]:
    block = "\n\n".join(
        f"[{summary.marker}] {summary.section}\n{summary.text}" for summary in summaries
    )
    return [
        {"role": "system", "content": REDUCE_SYSTEM},
        {"role": "user", "content": REDUCE_USER.format(summaries=block, question=question)},
    ]
