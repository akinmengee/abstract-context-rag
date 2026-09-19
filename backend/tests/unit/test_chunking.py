from abstractrag.core.config import ChunkingSettings
from abstractrag.rag.chunking.section_aware import SectionAwareChunker, embedding_text
from abstractrag.rag.models import BlockType
from tests.conftest import make_block, make_document


def chunker(**overrides) -> SectionAwareChunker:
    defaults = {"target_chars": 60, "overlap_chars": 10, "min_chunk_chars": 15}
    return SectionAwareChunker(ChunkingSettings(**{**defaults, **overrides}))


def test_chunks_never_span_two_sections():
    document = make_document(
        [
            make_block("Method text.", section=["Method"]),
            make_block("Results text.", section=["Results"]),
        ]
    )

    chunks = chunker().chunk(document)

    assert [chunk.metadata.section for chunk in chunks] == ["Method", "Results"]
    assert all(chunk.text.count("Results text.") <= 1 for chunk in chunks)


def test_reference_sections_are_dropped():
    document = make_document(
        [
            make_block("Body text that matters.", section=["Introduction"]),
            make_block("[1] Some citation.", section=["References"]),
        ]
    )

    chunks = chunker().chunk(document)

    assert len(chunks) == 1
    assert "citation" not in chunks[0].text


def test_headings_are_metadata_not_content():
    document = make_document(
        [
            make_block("Results", section=["Results"], block_type=BlockType.HEADING),
            make_block("We report 92.1 accuracy.", section=["Results"]),
        ]
    )

    chunks = chunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].text == "We report 92.1 accuracy."
    assert chunks[0].metadata.section == "Results"


def test_long_section_splits_and_carries_overlap():
    blocks = [make_block("a" * 40, section=["Method"]) for _ in range(3)]

    chunks = chunker(target_chars=50, overlap_chars=10).chunk(make_document(blocks))

    assert len(chunks) > 1
    assert chunks[1].text.startswith("a" * 10)


def test_short_tail_merges_into_previous_chunk():
    blocks = [make_block("a" * 60, section=["Method"]), make_block("tiny", section=["Method"])]

    chunks = chunker(target_chars=50, min_chunk_chars=20).chunk(make_document(blocks))

    assert len(chunks) == 1
    assert chunks[0].text.endswith("tiny")


def test_chunk_ids_are_stable_across_runs():
    document = make_document([make_block("Stable content.")])

    first = chunker().chunk(document)
    second = chunker().chunk(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_embedding_text_prefixes_the_section_breadcrumb():
    chunk = chunker().chunk(make_document([make_block("Body.", section=["Method"])]))[0]

    assert embedding_text(chunk).startswith("Test Paper > Method")
    assert chunk.text == "Body."
