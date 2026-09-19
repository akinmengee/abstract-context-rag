"""Section-aware chunking.

Chunks never span two sections, because a chunk that mixes "Method" and "Results"
embeds badly and cites worse. Within a section, blocks are packed up to a target
size with a small overlap so a sentence cut in half still appears whole somewhere.
Reference and acknowledgement sections are dropped here, where section names are known.
"""

from abstractrag.core.config import ChunkingSettings
from abstractrag.rag.models import (
    BlockType,
    Chunk,
    ChunkMetadata,
    DocumentBlock,
    ParsedDocument,
    chunk_id_for,
)

_DROPPED_SECTIONS = {
    "references",
    "reference",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "see also",
    "external links",
}


class SectionAwareChunker:
    def __init__(self, settings: ChunkingSettings) -> None:
        self.settings = settings

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section_path, blocks in self._sections(document):
            for text, pages, block_types in self._split(blocks):
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id_for(document.document_id, index),
                        document_id=document.document_id,
                        index=index,
                        text=text,
                        metadata=ChunkMetadata(
                            document_id=document.document_id,
                            title=document.title,
                            source_type=document.source_type,
                            origin=document.origin,
                            section=section_path[-1] if section_path else None,
                            section_path=list(section_path),
                            page=min(pages) if pages else None,
                            block_types=block_types,
                        ),
                    )
                )
        return chunks

    def _sections(self, document: ParsedDocument):
        """Group consecutive blocks that share a section path, dropping noise sections."""
        current_path: list[str] | None = None
        current_blocks: list[DocumentBlock] = []

        for block in document.blocks:
            if _is_dropped(block.section_path):
                continue
            # The heading itself is context, not content; it is carried in the metadata.
            if block.block_type is BlockType.HEADING:
                continue

            if block.section_path != current_path:
                if current_blocks:
                    yield current_path or [], current_blocks
                current_path, current_blocks = block.section_path, []
            current_blocks.append(block)

        if current_blocks:
            yield current_path or [], current_blocks

    def _split(self, blocks: list[DocumentBlock]):
        """Pack blocks into target-sized pieces, prefixing each with the previous tail."""
        pieces = self._group(blocks)
        previous_text = ""

        for piece in pieces:
            text = "\n\n".join(block.text for block in piece).strip()
            if previous_text and self.settings.overlap_chars:
                # The overlap keeps a sentence that was cut at a boundary readable
                # in at least one chunk.
                text = f"{previous_text[-self.settings.overlap_chars :]}\n\n{text}"
            previous_text = text

            pages = sorted({block.page for block in piece if block.page is not None})
            block_types = sorted({block.block_type for block in piece})
            yield text, pages, block_types

    def _group(self, blocks: list[DocumentBlock]) -> list[list[DocumentBlock]]:
        """Split a section's blocks at the target size, merging a too-short tail back."""
        pieces: list[list[DocumentBlock]] = []
        current: list[DocumentBlock] = []
        size = 0

        for block in blocks:
            current.append(block)
            size += len(block.text)
            if size >= self.settings.target_chars:
                pieces.append(current)
                current, size = [], 0

        if current:
            if pieces and size < self.settings.min_chunk_chars:
                pieces[-1].extend(current)
            else:
                pieces.append(current)
        return pieces


def _is_dropped(section_path: list[str]) -> bool:
    return any(part.strip().lower() in _DROPPED_SECTIONS for part in section_path)


def embedding_text(chunk: Chunk) -> str:
    """What actually gets embedded: the section breadcrumb plus the chunk text.

    The breadcrumb gives an isolated chunk enough context to match a query that
    names the section ("what does the ablation say"). Chunk.text stays untouched
    so citations and verification always quote the source.
    """
    breadcrumb = " > ".join([chunk.metadata.title, *chunk.metadata.section_path])
    return f"{breadcrumb}\n\n{chunk.text}"
