"""Map-reduce summarisation: the pipeline top-k retrieval cannot replace.

"Summarise this paper" is a global question - the answer is spread across every
section, so retrieving five chunks can only ever miss most of it. Map summarises
each section from its real text; reduce writes the answer from those summaries,
citing them by number.

The cost is the honest trade-off: one LLM call per section plus one, where a
normal answer costs one (rag.md section 8.1). With summarization.method raptor
the map stage is replaced by a tree built at indexing time, and only reduce()
runs per question (rag.md 7.9.1).
"""

from dataclasses import dataclass

from abstractrag.core.config import SummarizationSettings
from abstractrag.core.errors import LLMError
from abstractrag.core.logging import get_logger
from abstractrag.rag.generation.llm_client import LlamaCppClient
from abstractrag.rag.generation.prompts import CITATION_MARKER
from abstractrag.rag.models import Chunk, Citation, SectionSummary
from abstractrag.rag.summarization import prompts
from abstractrag.rag.summarization.sections import SectionGroup, group_sections

logger = get_logger(__name__)


@dataclass
class MapReduceSummarizer:
    llm: LlamaCppClient
    settings: SummarizationSettings

    def summarize(self, question: str, chunks: list[Chunk]) -> tuple[str, list[SectionSummary]]:
        """Summarise a whole document: the text, and what its markers refer to."""
        groups = group_sections(chunks, self.settings.max_group_chars)
        summaries = self._map(question, groups)
        if not summaries:
            return "", []
        return self.reduce(question, summaries), summaries

    def _map(self, question: str, groups: list[SectionGroup]) -> list[SectionSummary]:
        """One call per group, logged as it goes - this is minutes, not seconds.

        Sections with nothing to say are dropped and markers are numbered over
        what survives, so the reduce stage never sees a gap it has to explain.
        A section whose LLM call fails is dropped the same way: losing one
        section beats discarding every already-completed call in the run.
        """
        summaries: list[SectionSummary] = []
        for number, group in enumerate(groups, start=1):
            logger.info("summarising section %d/%d: %s", number, len(groups), group.section)
            try:
                reply = self.llm.complete(prompts.build_map_messages(question, group))
            except LLMError as exc:
                logger.warning("skipping section %r after LLM failure: %s", group.section, exc)
                continue
            # Inline references like "[12]" in the source text would otherwise be
            # indistinguishable from a genuine section marker in the reduce prompt.
            text = CITATION_MARKER.sub("", reply).strip()
            if not text or prompts.NOTHING_RELEVANT in text:
                continue
            summaries.append(
                SectionSummary(
                    marker=len(summaries) + 1,
                    section=group.section,
                    text=text,
                    chunk_ids=group.chunk_ids,
                )
            )
        return summaries

    def reduce(self, question: str, summaries: list[SectionSummary]) -> str:
        """Write the answer from numbered section summaries - one LLM call."""
        # No hard cap (finding 2 asked for visibility, not truncation) - just a
        # log if the assembled summaries are large enough to risk the context
        # window, since an overflow there truncates silently, not with an error.
        total_chars = sum(len(summary.section) + len(summary.text) for summary in summaries)
        threshold = self.settings.max_group_chars * 4
        if total_chars > threshold:
            logger.warning(
                "reduce prompt is large: %d chars across %d summaries (over the "
                "%d-char guideline derived from max_group_chars) - risks silently "
                "overflowing the LLM's context window",
                total_chars,
                len(summaries),
                threshold,
            )
        return self.llm.complete(prompts.build_reduce_messages(question, summaries)).strip()


def citations_for(
    text: str, summaries: list[SectionSummary], chunks: list[Chunk]
) -> list[Citation]:
    """Citations for the markers the summary actually used.

    A summary's marker means a whole section, so `chunk_id` is that section's
    first chunk and the full list stays on `SectionSummary.chunk_ids`.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    used = {int(marker) for marker in CITATION_MARKER.findall(text)}

    citations: list[Citation] = []
    for summary in summaries:
        first = next((by_id[cid] for cid in summary.chunk_ids if cid in by_id), None)
        if summary.marker not in used or first is None:
            continue
        citations.append(
            Citation(
                marker=summary.marker,
                chunk_id=first.chunk_id,
                title=first.metadata.title,
                section=summary.section,
                page=first.metadata.page,
                origin=first.metadata.origin,
            )
        )
    return citations


def passages_for(summaries: list[SectionSummary], chunks: list[Chunk]) -> dict[int, str]:
    """Marker -> the section's real source text.

    Deliberately not the section summary: a claim invented while summarising a
    section would look supported when checked against the text that invented it.
    """
    texts = {chunk.chunk_id: chunk.text for chunk in chunks}
    return {
        summary.marker: "\n\n".join(
            texts[chunk_id] for chunk_id in summary.chunk_ids if chunk_id in texts
        )
        for summary in summaries
    }
