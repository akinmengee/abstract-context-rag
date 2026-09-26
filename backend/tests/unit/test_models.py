from abstractrag.rag.models import BlockType, ChunkMetadata, DocumentBlock, SourceType


def test_figure_is_a_block_type():
    assert BlockType.FIGURE.value == "figure"


def test_document_block_image_path_defaults_to_none():
    block = DocumentBlock(text="a figure caption")
    assert block.image_path is None


def test_chunk_metadata_image_paths_defaults_to_empty():
    metadata = ChunkMetadata(document_id="doc", title="t", source_type=SourceType.PDF, origin="o")
    assert metadata.image_paths == []
