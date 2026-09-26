"""Shared test factories."""

from abstractrag.rag.models import (
    BlockType,
    Chunk,
    ChunkMetadata,
    DocumentBlock,
    ParsedDocument,
    SourceType,
    chunk_id_for,
    document_id_for,
)


def make_block(
    text: str,
    section: list[str] | None = None,
    page: int = 1,
    block_type: BlockType = BlockType.PARAGRAPH,
    image_path: str | None = None,
) -> DocumentBlock:
    return DocumentBlock(
        text=text,
        block_type=block_type,
        section_path=section or ["Introduction"],
        page=page,
        image_path=image_path,
    )


def make_document(blocks: list[DocumentBlock], title: str = "Test Paper") -> ParsedDocument:
    origin = f"test:{title}"
    return ParsedDocument(
        document_id=document_id_for(origin),
        title=title,
        source_type=SourceType.PDF,
        origin=origin,
        blocks=blocks,
    )


def make_chunk(
    text: str,
    index: int = 0,
    section: str | None = "Results",
    section_path: list[str] | None = None,
    image_paths: list[str] | None = None,
) -> Chunk:
    document_id = document_id_for("test:chunk")
    default_path = [section] if section else []
    return Chunk(
        chunk_id=chunk_id_for(document_id, index),
        document_id=document_id,
        index=index,
        text=text,
        metadata=ChunkMetadata(
            document_id=document_id,
            title="Test Paper",
            source_type=SourceType.PDF,
            origin="test:chunk",
            section=section,
            section_path=section_path if section_path is not None else default_path,
            page=index + 1,
            image_paths=image_paths or [],
        ),
    )
