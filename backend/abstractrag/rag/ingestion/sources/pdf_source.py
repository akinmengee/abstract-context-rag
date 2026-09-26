"""PDF -> ParsedDocument via Docling layout analysis.

Layout analysis (not raw text extraction) is what makes two-column papers work:
Docling labels every block and emits them in reading order, so the left column is
never interleaved with the right one. Page headers, footers and images are dropped
here because they are parser-level noise; reference sections are dropped later, at
chunking, where section names are known.

Figures are the one exception (rag.md 7.9.3/7.9.4): when
ingestion.picture_description is enabled, Docling's do_picture_description
step describes each one with a vision model served by Ollama at ingest time.
When ingestion.figures is enabled (independently - the two do not need each
other), each figure's cropped image is instead saved to disk and its path
carried onto the block, so a vision model can be asked about it later, with
the user's actual question, at answer time. Both disabled (the default), a
picture still has no text and is dropped exactly as before either existed.
"""

from pathlib import Path
from typing import Any

from abstractrag.core.config import FigureExtractionSettings, PictureDescriptionSettings
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
    "document_index",
    "form",
    "key_value_region",
    "checkbox_selected",
    "checkbox_unselected",
}


def figure_image_path(figures_dir: Path, document_id: str, index: int) -> Path:
    """Where one document's Nth figure image is saved - deterministic, so a
    re-ingest overwrites the same files instead of accumulating orphans."""
    return figures_dir / document_id / f"{index}.png"


class PdfSource:
    """Source adapter for PDFs, whether uploaded by the user or downloaded from arXiv."""

    def __init__(
        self,
        picture_description: PictureDescriptionSettings | None = None,
        llm_base_url: str = "",
        figures: FigureExtractionSettings | None = None,
        figures_dir: Path | None = None,
    ) -> None:
        self._converter: Any | None = None
        self.picture_description = picture_description or PictureDescriptionSettings()
        self.llm_base_url = llm_base_url
        self.figures = figures or FigureExtractionSettings()
        self.figures_dir = figures_dir or Path("data/documents/figures")

    @property
    def converter(self) -> Any:
        # Imported and constructed on first use: docling pulls in the layout models.
        if self._converter is None:
            self._converter = self._build_converter()
        return self._converter

    def _build_converter(self) -> Any:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            PictureDescriptionApiOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        if not self.picture_description.enabled and not self.figures.enabled:
            return DocumentConverter()

        options = PdfPipelineOptions(generate_picture_images=True)
        if self.picture_description.enabled:
            options.do_picture_description = True
            # Docling treats any API-based picture-description backend as a
            # "remote service" and refuses it otherwise, even though ours is
            # localhost - this is Docling's opt-in for calling out at all, not
            # a claim about where the endpoint lives.
            options.enable_remote_services = True
            options.picture_description_options = PictureDescriptionApiOptions(
                url=f"{self.llm_base_url}/chat/completions",
                params={"model": self.picture_description.model},
                prompt=self.picture_description.prompt,
                timeout=self.picture_description.timeout_seconds,
            )
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

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

        document_id = document_id_for(origin)
        blocks = list(self._to_blocks(document, document_id))
        if not blocks:
            raise ParseError(f"no readable blocks in {pdf_path.name}")

        resolved_title = title or _first_title(blocks) or pdf_path.stem
        logger.info("parsed %s: %d blocks", pdf_path.name, len(blocks))

        return ParsedDocument(
            document_id=document_id,
            title=resolved_title,
            source_type=SourceType.PDF,
            origin=origin,
            blocks=blocks,
            extra=extra or {},
        )

    def _to_blocks(self, document: Any, document_id: str):
        section_path: list[str] = []
        figure_index = 0

        for item, _tree_level in document.iterate_items():
            label = _label_of(item)
            if label in _SKIP_LABELS:
                continue

            image_path: str | None = None
            if label == "picture":
                block_type = BlockType.FIGURE
                text = _figure_description(item)
                if self.figures.enabled:
                    saved = self._save_figure(item, document, document_id, figure_index)
                    if saved is not None:
                        image_path = str(saved)
                        figure_index += 1
                if not text and image_path is None:
                    continue
            else:
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
                image_path=image_path,
            )

    def _save_figure(
        self, item: Any, document: Any, document_id: str, index: int
    ) -> Path | None:
        """Crop and save one figure; None (not an empty file) when Docling has
        no image for it - generate_picture_images can still miss one."""
        image = item.get_image(document)
        if image is None:
            return None
        path = figure_image_path(self.figures_dir, document_id, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path


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


def _figure_description(item: Any) -> str:
    """The vision model's caption, from Docling's do_picture_description step.

    Empty (and so dropped, same as before this feature existed) when picture
    description is disabled, or the model returned nothing for this figure.
    """
    for annotation in getattr(item, "annotations", []):
        if getattr(annotation, "kind", "") == "description":
            return str(getattr(annotation, "text", "")).strip()
    return ""


def _page_of(item: Any) -> int | None:
    provenance = getattr(item, "prov", None)
    if not provenance:
        return None
    return getattr(provenance[0], "page_no", None)


def _first_title(blocks: list[DocumentBlock]) -> str | None:
    return next((block.text for block in blocks if block.block_type is BlockType.TITLE), None)
