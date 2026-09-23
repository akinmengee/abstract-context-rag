"""Grouping a document's chunks into map units, without an LLM.

Map-reduce summarises one group per call, so grouping sets both the cost (how
many calls) and the meaning of a citation (what "[2]" points at). Top-level
sections are the unit: the document's own structure, and few enough that a
summary costs roughly one call per section instead of one per subsection.
"""

from dataclasses import dataclass, field

from abstractrag.rag.models import Chunk

UNTITLED = "Document"


@dataclass
class SectionGroup:
    """One map call's worth of text: a section, or part of an oversized one."""

    section: str
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(chunk.text for chunk in self.chunks)

    @property
    def chunk_ids(self) -> list[str]:
        return [chunk.chunk_id for chunk in self.chunks]


def group_sections(chunks: list[Chunk], max_chars: int) -> list[SectionGroup]:
    """Consecutive chunks of one top-level section, split to fit a single call.

    Runs are consecutive rather than keyed by name, so the summary follows the
    document's reading order even if a section name reappears later. A chunk
    longer than the budget still gets its own group: splitting inside a chunk
    would cut a passage that chunking deliberately kept whole.
    """
    groups: list[SectionGroup] = []
    for chunk in sorted(chunks, key=lambda item: item.index):
        section = _top_level(chunk)
        current = groups[-1] if groups else None
        if current is None or current.section != section or _overflows(current, chunk, max_chars):
            groups.append(SectionGroup(section=section, chunks=[chunk]))
        else:
            current.chunks.append(chunk)
    return groups


def _top_level(chunk: Chunk) -> str:
    metadata = chunk.metadata
    if metadata.section_path:
        return metadata.section_path[0]
    return metadata.section or UNTITLED


def _overflows(group: SectionGroup, chunk: Chunk, max_chars: int) -> bool:
    return len(group.text) + len(chunk.text) > max_chars
