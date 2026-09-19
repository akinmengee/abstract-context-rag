"""Wikipedia article -> ParsedDocument.

No layout analysis needed: the extract is already clean text where `== Heading ==`
marks the structure, so the adapter only has to rebuild the heading hierarchy.
"""

import re

from abstractrag.core.errors import ParseError
from abstractrag.rag.ingestion.fetchers.wikipedia import WikipediaArticle
from abstractrag.rag.models import (
    BlockType,
    DocumentBlock,
    ParsedDocument,
    SourceType,
    document_id_for,
)

_HEADING = re.compile(r"^(?P<level>={2,6})\s*(?P<title>.+?)\s*(?P=level)$")

# Navigational sections that carry no content worth retrieving.
_SKIP_SECTIONS = {
    "see also",
    "references",
    "external links",
    "further reading",
    "notes",
    "bibliography",
    "sources",
    "citations",
}


class WikipediaSource:
    """Source adapter for single Wikipedia articles."""

    def parse(self, article: WikipediaArticle) -> ParsedDocument:
        blocks = list(self._to_blocks(article.text))
        if not blocks:
            raise ParseError(f"no readable content in Wikipedia article {article.title!r}")

        return ParsedDocument(
            document_id=document_id_for(article.url),
            title=article.title,
            source_type=SourceType.WIKIPEDIA,
            origin=article.url,
            blocks=blocks,
            extra={"revision_id": article.revision_id},
        )

    def _to_blocks(self, text: str):
        # Line based, because the extract separates a heading from its first
        # paragraph with a single newline in some articles and a blank line in others.
        section_path: list[str] = []
        skipping = False
        buffer: list[str] = []

        def take_paragraph() -> str:
            paragraph = " ".join(buffer).strip()
            buffer.clear()
            return paragraph

        for raw_line in text.splitlines():
            line = raw_line.strip()
            heading = _HEADING.match(line)

            if heading or not line:
                paragraph = take_paragraph()
                if paragraph and not skipping:
                    yield DocumentBlock(
                        text=paragraph,
                        block_type=BlockType.PARAGRAPH,
                        section_path=list(section_path),
                    )

            if not heading:
                if line:
                    buffer.append(line)
                continue

            # "==" is a top-level section, "===" a subsection, and so on.
            depth = len(heading.group("level")) - 1
            title = heading.group("title")
            section_path = section_path[: depth - 1] + [title]
            skipping = title.lower() in _SKIP_SECTIONS
            if not skipping:
                yield DocumentBlock(
                    text=title,
                    block_type=BlockType.HEADING,
                    section_path=list(section_path),
                )

        paragraph = take_paragraph()
        if paragraph and not skipping:
            yield DocumentBlock(
                text=paragraph,
                block_type=BlockType.PARAGRAPH,
                section_path=list(section_path),
            )
