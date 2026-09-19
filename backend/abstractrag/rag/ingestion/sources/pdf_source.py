"""PDF -> ParsedDocument via Docling layout analysis.

Layout analysis (not raw text extraction) is what makes two-column papers work:
Docling labels every block and emits them in reading order, so the left column is
never interleaved with the right one. Page headers, footers and images are dropped
here because they are parser-level noise; reference sections are dropped later, at
chunking, where section names are known.
"""

from pathlib import Path
from typing import Any

from abstractrag.core.errors import ParseError
from abstractrag.core.logging import get_logger
from abstractrag.rag.models import (
    BlockType,
    DocumentBlock,
    ParsedDocument,
    SourceType,
    document_id_for,
)

logger = get_logger(__name__)

# Matched on the label's string value so a docling version bump cannot break imports.
_LABEL_TO_BLOCK = {
    "title": BlockType.TITLE,
    "section_header": BlockType.HEADING,
    "text": BlockType.PARAGRAPH,
    "paragraph": BlockType.PARAGRAPH,
    "list_item": BlockType.LIST,
    "table": BlockType.TABLE,
    "caption": BlockType.CAPTION,
    "formula": BlockType.FORMULA,
}

_SKIP_LABELS = {
    "page_header",
    "page_footer",
    "footnote",
    "picture",
    "document_index",
    "form",
    "key_value_region",
    "checkbox_selected",
    "checkbox_unselected",
}


class PdfSource:
    """Source adapter for PDFs, whether uploaded by the user or downloaded from arXiv."""

    def __init__(self) -> None:
        self._converter: Any | None = None

    @property
    def converter(self) -> Any:
        # Imported and constructed on first use: docling pulls in the layout models.
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def parse(
        self,
        pdf_path: Path,
        origin: str,
        title: str | None = None,
        extra: dict[str, str] | None = None,
    ) -> ParsedDocument:
        try:
            document = self.converter.convert(pdf_path).document
        except Exception as exc:
            raise ParseError(f"docling failed on {pdf_path.name}: {exc}") from exc

        blocks = list(self._to_blocks(document))
        if not blocks:
            raise ParseError(f"no readable blocks in {pdf_path.name}")

        resolved_title = title or _first_title(blocks) or pdf_path.stem
        logger.info("parsed %s: %d blocks", pdf_path.name, len(blocks))

        return ParsedDocument(
            document_id=document_id_for(origin),
            title=resolved_title,
            source_type=SourceType.PDF,
            origin=origin,
            blocks=blocks,
            extra=extra or {},
        )

    def _to_blocks(self, document: Any):
        section_path: list[str] = []

        for item, _tree_level in document.iterate_items():
            label = _label_of(item)
            if label in _SKIP_LABELS:
                continue

            block_type = _LABEL_TO_BLOCK.get(label, BlockType.OTHER)
            text = _text_of(item, document)
            if not text:
                continue

            if block_type is BlockType.HEADING:
                heading_level = max(int(getattr(item, "level", 1) or 1), 1)
                section_path = section_path[: heading_level - 1] + [text]

            yield DocumentBlock(
                text=text,
                block_type=block_type,
                section_path=list(section_path),
                page=_page_of(item),
            )


def _label_of(item: Any) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label)).lower()


def _text_of(item: Any, document: Any) -> str:
    """Tables are exported as markdown so numbers survive; everything else is plain text."""
    if hasattr(item, "export_to_markdown"):
        try:
            return item.export_to_markdown(document).strip()
        except TypeError:
            # Older docling versions take no document argument.
            return item.export_to_markdown().strip()
    return str(getattr(item, "text", "")).strip()


def _page_of(item: Any) -> int | None:
    provenance = getattr(item, "prov", None)
    if not provenance:
        return None
    return getattr(provenance[0], "page_no", None)


def _first_title(blocks: list[DocumentBlock]) -> str | None:
    return next((block.text for block in blocks if block.block_type is BlockType.TITLE), None)
