"""Map-reduce summarisation: grouping sections, then summarising them."""

from abstractrag.core.config import EmbeddingSettings, QdrantSettings
from abstractrag.database.qdrant_store import QdrantStore
from abstractrag.rag.models import Answer, SectionSummary
from abstractrag.rag.summarization.sections import group_sections
from tests.conftest import make_chunk


class TestGroupSections:
    def test_subsections_collapse_into_their_top_level_section(self):
        # The whole point of grouping: ~18 subsection calls become ~6 section calls.
        chunks = [
            make_chunk("a", index=0, section="2.1 Setup", section_path=["2 Methods", "2.1 Setup"]),
            make_chunk("b", index=1, section="2.2 DPR", section_path=["2 Methods", "2.2 DPR"]),
            make_chunk("c", index=2, section="3.1 QA", section_path=["3 Experiments", "3.1 QA"]),
        ]

        groups = group_sections(chunks, max_chars=1000)

        assert [group.section for group in groups] == ["2 Methods", "3 Experiments"]
        assert [len(group.chunks) for group in groups] == [2, 1]

    def test_chunks_are_grouped_in_document_order_not_arrival_order(self):
        chunks = [
            make_chunk("second", index=1, section_path=["1 Intro"]),
            make_chunk("first", index=0, section_path=["1 Intro"]),
        ]

        assert group_sections(chunks, max_chars=1000)[0].text == "first\n\nsecond"

    def test_an_oversized_section_splits_but_keeps_its_name(self):
        # Each part is still "3 Experiments": a citation must name the section,
        # not an arbitrary slice number.
        chunks = [make_chunk("x" * 400, index=i, section_path=["3 Experiments"]) for i in range(3)]

        groups = group_sections(chunks, max_chars=900)

        assert [group.section for group in groups] == ["3 Experiments", "3 Experiments"]
        assert [len(group.chunks) for group in groups] == [2, 1]

    def test_a_single_chunk_over_budget_still_forms_one_group(self):
        # Splitting inside a chunk would cut a passage chunking kept whole on purpose.
        chunks = [make_chunk("x" * 5000, index=0, section_path=["1 Intro"])]

        assert len(group_sections(chunks, max_chars=100)) == 1

    def test_a_section_name_that_reappears_later_starts_a_new_group(self):
        # Runs are consecutive, so the summary follows reading order.
        chunks = [
            make_chunk("a", index=0, section_path=["1 Intro"]),
            make_chunk("b", index=1, section_path=["2 Methods"]),
            make_chunk("c", index=2, section_path=["1 Intro"]),
        ]

        assert [group.section for group in group_sections(chunks, max_chars=1000)] == [
            "1 Intro",
            "2 Methods",
            "1 Intro",
        ]

    def test_a_chunk_without_a_section_path_falls_back_to_its_section(self):
        chunks = [make_chunk("a", index=0, section="Abstract", section_path=[])]

        assert group_sections(chunks, max_chars=1000)[0].section == "Abstract"

    def test_a_chunk_with_no_section_at_all_is_still_grouped(self):
        chunks = [make_chunk("a", index=0, section=None, section_path=[])]

        assert group_sections(chunks, max_chars=1000)[0].section == "Document"

    def test_no_chunks_means_no_groups(self):
        assert group_sections([], max_chars=1000) == []

    def test_a_group_exposes_the_chunk_ids_it_covers(self):
        # Verification resolves a marker back to these, so they must be complete.
        chunks = [make_chunk("a", index=0), make_chunk("b", index=1)]

        group = group_sections(chunks, max_chars=1000)[0]

        assert group.chunk_ids == [chunks[0].chunk_id, chunks[1].chunk_id]


class _FakePoint:
    def __init__(self, chunk):
        self.payload = {"chunk": chunk.model_dump(mode="json")}


class _FakeClient:
    """Scrolls one page at a time so the paging loop is actually exercised."""

    def __init__(self, chunks, page_size=2):
        self.chunks = chunks
        self.page_size = page_size
        self.filters = []

    def scroll(
        self, collection_name, scroll_filter=None, limit=None, offset=None, with_payload=True
    ):
        self.filters.append(scroll_filter)
        start = offset or 0
        page = self.chunks[start : start + self.page_size]
        next_offset = start + self.page_size if start + self.page_size < len(self.chunks) else None
        return [_FakePoint(chunk) for chunk in page], next_offset


def _store(chunks):
    store = QdrantStore(QdrantSettings(), EmbeddingSettings())
    store.client = _FakeClient(chunks)
    return store


class TestListChunks:
    def test_every_page_is_read_not_just_the_first(self):
        chunks = [make_chunk(f"c{i}", index=i) for i in range(5)]

        assert len(_store(chunks).list_chunks("doc-1")) == 5

    def test_chunks_come_back_in_document_order(self):
        chunks = [make_chunk("second", index=1), make_chunk("first", index=0)]

        assert [chunk.text for chunk in _store(chunks).list_chunks("doc-1")] == ["first", "second"]

    def test_the_scroll_is_filtered_to_one_document(self):
        # Without the filter this would summarise every ingested paper at once.
        store = _store([make_chunk("a", index=0)])

        store.list_chunks("doc-1")

        assert store.client.filters and store.client.filters[0] is not None

    def test_an_unknown_document_yields_nothing(self):
        assert _store([]).list_chunks("doc-1") == []


class TestSummaryModels:
    def test_an_answer_carries_no_section_summaries_by_default(self):
        # A normal retrieval answer has no sections; the field must stay optional.
        assert Answer(text="x").section_summaries == []

    def test_a_section_summary_keeps_every_chunk_it_covers(self):
        # Verification resolves a marker to these chunks, so a partial list would
        # silently narrow what a claim is checked against.
        summary = SectionSummary(marker=1, section="2 Methods", text="s", chunk_ids=["a", "b"])

        assert summary.chunk_ids == ["a", "b"]
