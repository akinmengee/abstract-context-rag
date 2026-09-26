from types import SimpleNamespace

import pytest

from abstractrag.core.errors import InvalidInputError, SourceNotFoundError
from abstractrag.rag.ingestion.base import SourceInput
from abstractrag.rag.ingestion.fetchers.arxiv import normalize_arxiv_id
from abstractrag.rag.ingestion.fetchers.wikipedia import (
    WikipediaArticle,
    normalize_article_title,
)
from abstractrag.rag.ingestion.sources.pdf_source import _figure_description, figure_image_path
from abstractrag.rag.ingestion.sources.wikipedia_source import WikipediaSource
from abstractrag.rag.models import BlockType, SourceType


class TestSourceInput:
    def test_accepts_exactly_one_source(self):
        assert SourceInput(arxiv_id="2005.11401").describe() == "2005.11401"

    def test_rejects_no_source(self):
        with pytest.raises(InvalidInputError):
            SourceInput()

    def test_rejects_two_sources(self):
        with pytest.raises(InvalidInputError):
            SourceInput(arxiv_id="2005.11401", wikipedia="RAG")

    def test_upload_requires_a_file_name(self):
        with pytest.raises(InvalidInputError):
            SourceInput(file_bytes=b"%PDF-1.4")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2005.11401", "2005.11401"),
        ("arXiv:2005.11401v2", "2005.11401v2"),
        ("https://arxiv.org/abs/2004.04906", "2004.04906"),
        ("https://arxiv.org/pdf/1706.03762v7", "1706.03762v7"),
    ],
)
def test_arxiv_ids_are_normalized(raw, expected):
    assert normalize_arxiv_id(raw) == expected


def test_invalid_arxiv_id_is_rejected():
    with pytest.raises(SourceNotFoundError):
        normalize_arxiv_id("not-a-paper")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Retrieval-augmented generation", "Retrieval-augmented generation"),
        (
            "https://en.wikipedia.org/wiki/Transformer_(deep_learning)",
            "Transformer (deep learning)",
        ),
    ],
)
def test_wikipedia_titles_are_normalized(raw, expected):
    assert normalize_article_title(raw) == expected


class TestWikipediaSource:
    article = WikipediaArticle(
        title="Example",
        url="https://en.wikipedia.org/wiki/Example",
        revision_id="42",
        text=(
            "Lead paragraph.\n\n"
            "== History ==\n\n"
            "It started early.\n\n"
            "=== Early years ===\n\n"
            "Details about the early years.\n\n"
            "== References ==\n\n"
            "[1] Some citation."
        ),
    )

    def test_builds_the_heading_hierarchy(self):
        document = WikipediaSource().parse(self.article)
        paragraphs = [block for block in document.blocks if block.block_type is BlockType.PARAGRAPH]

        assert document.source_type is SourceType.WIKIPEDIA
        assert paragraphs[0].section_path == []
        assert paragraphs[1].section_path == ["History"]
        assert paragraphs[2].section_path == ["History", "Early years"]

    def test_drops_navigational_sections(self):
        document = WikipediaSource().parse(self.article)

        assert all("citation" not in block.text for block in document.blocks)

    def test_keeps_the_revision_id_for_citations(self):
        document = WikipediaSource().parse(self.article)

        assert document.extra["revision_id"] == "42"
        assert document.origin == self.article.url


class TestFigureDescription:
    def test_reads_the_description_annotation(self):
        item = SimpleNamespace(
            annotations=[SimpleNamespace(kind="description", text=" A bar chart of accuracy. ")]
        )
        assert _figure_description(item) == "A bar chart of accuracy."

    def test_empty_without_annotations(self):
        assert _figure_description(SimpleNamespace(annotations=[])) == ""

    def test_empty_when_the_item_has_no_annotations_attribute(self):
        assert _figure_description(SimpleNamespace()) == ""

    def test_ignores_annotations_of_a_different_kind(self):
        item = SimpleNamespace(annotations=[SimpleNamespace(kind="classification", text="chart")])
        assert _figure_description(item) == ""


class TestFigureImagePath:
    def test_builds_a_path_under_the_documents_directory(self, tmp_path):
        path = figure_image_path(tmp_path, "doc-123", 0)
        assert path == tmp_path / "doc-123" / "0.png"

    def test_different_indices_do_not_collide(self, tmp_path):
        first = figure_image_path(tmp_path, "doc-123", 0)
        second = figure_image_path(tmp_path, "doc-123", 1)
        assert first != second
