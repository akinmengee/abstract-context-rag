"""RAPTOR: clustering chunks and building the summary tree over them."""

import math

from abstractrag.core.config import RaptorSettings
from abstractrag.core.errors import LLMError
from abstractrag.rag.embedding.bge_m3 import Embedding
from abstractrag.rag.raptor.clustering import cluster
from abstractrag.rag.raptor.tree import TreeBuilder
from tests.conftest import make_chunk


def blob(center: list[float], count: int, spread: float = 0.01) -> list[list[float]]:
    """Points close to one direction, deterministic."""
    return [[c + spread * ((i + j) % 3) for j, c in enumerate(center)] for i in range(count)]


class TestCluster:
    def test_nothing_to_cluster(self):
        assert cluster([], cluster_size=5) == []

    def test_a_small_set_is_one_cluster(self):
        assert cluster(blob([1.0, 0.0], 4), cluster_size=5) == [[0, 1, 2, 3]]

    def test_separate_topics_land_in_separate_clusters(self):
        vectors = blob([1.0, 0.0, 0.0], 3) + blob([0.0, 1.0, 0.0], 3)

        assert cluster(vectors, cluster_size=3) == [[0, 1, 2], [3, 4, 5]]

    def test_every_index_appears_exactly_once(self):
        vectors = [[math.sin(i), math.cos(i), math.sin(2 * i)] for i in range(23)]

        groups = cluster(vectors, cluster_size=4)

        assert sorted(i for group in groups for i in group) == list(range(23))

    def test_no_cluster_grows_past_twice_the_target_size(self):
        # An oversized cluster would not fit one summarisation prompt.
        vectors = blob([1.0, 0.0, 0.0], 20) + blob([0.0, 1.0, 0.0], 2)

        assert max(len(group) for group in cluster(vectors, cluster_size=3)) <= 6

    def test_is_deterministic(self):
        vectors = [[math.sin(i), math.cos(i)] for i in range(17)]

        assert cluster(vectors, cluster_size=4) == cluster(vectors, cluster_size=4)


class TopicEmbedder:
    """Puts every text whose body starts with "a" on one axis, "b" on another."""

    def embed(self, texts: list[str]) -> list[Embedding]:
        return [Embedding(dense=self._vector(text), sparse={}) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        body = text.split("\n\n", 1)[-1]
        return [1.0, 0.0] if body.startswith("a") else [0.0, 1.0]


class SummaryLlm:
    """Summarises by echoing a fixed reply; can fail on chosen calls."""

    def __init__(self, reply: str = "a summary", fail_on: set[int] | None = None) -> None:
        self.reply = reply
        self.fail_on = fail_on or set()
        self.calls = 0

    def complete(self, messages) -> str:
        self.calls += 1
        if self.calls in self.fail_on:
            raise LLMError("boom")
        return self.reply


def leaves(*texts: str, sections: list[str] | None = None) -> list:
    names = sections or [f"S{i}" for i in range(len(texts))]
    return [make_chunk(text, index=i, section=names[i]) for i, text in enumerate(texts)]


def builder(llm=None, **settings) -> TreeBuilder:
    return TreeBuilder(
        llm=llm or SummaryLlm(),
        embedder=TopicEmbedder(),
        settings=RaptorSettings(**{"cluster_size": 2, **settings}),
    )


def nodes_of(pairs):
    return [node for node, _ in pairs]


class TestTreeBuilder:
    def test_builds_levels_until_one_root(self):
        nodes = nodes_of(builder().build(leaves("a1", "a2", "b1", "b2"), lambda: None))

        assert [node.metadata.level for node in nodes] == [1, 1, 2]

    def test_a_node_covers_the_leaves_below_it_not_the_nodes(self):
        chunks = leaves("a1", "a2", "b1", "b2")

        level_one_a, _, root = nodes_of(builder().build(chunks, lambda: None))

        assert level_one_a.metadata.source_ids == [chunks[0].chunk_id, chunks[1].chunk_id]
        assert root.metadata.source_ids == [chunk.chunk_id for chunk in chunks]

    def test_a_node_is_labelled_with_the_sections_it_summarises(self):
        chunks = leaves("a1", "a2", "b1", "b2", sections=["Intro", "Intro", "Method", "Results"])

        nodes = nodes_of(builder().build(chunks, lambda: None))

        assert nodes[0].metadata.section == "Intro"
        assert nodes[1].metadata.section == "Method; Results"

    def test_node_ids_are_stable_and_distinct_from_leaf_ids(self):
        chunks = leaves("a1", "a2", "b1", "b2")

        first = [n.chunk_id for n in nodes_of(builder().build(chunks, lambda: None))]
        second = [n.chunk_id for n in nodes_of(builder().build(chunks, lambda: None))]

        assert first == second
        assert not set(first) & {chunk.chunk_id for chunk in chunks}

    def test_stops_at_the_level_limit(self):
        nodes = nodes_of(
            builder(max_levels=1).build(leaves("a1", "a2", "b1", "b2"), lambda: None)
        )

        assert {node.metadata.level for node in nodes} == {1}

    def test_a_single_chunk_needs_no_tree(self):
        assert builder().build(leaves("a1"), lambda: None) == []

    def test_a_failed_summary_drops_that_cluster_only(self):
        # Losing one cluster beats losing the whole tree (same rule as map-reduce).
        nodes = nodes_of(
            builder(llm=SummaryLlm(fail_on={1}), max_levels=1).build(
                leaves("a1", "a2", "b1", "b2"), lambda: None
            )
        )

        assert len(nodes) == 1

    def test_inline_reference_markers_are_stripped_from_summaries(self):
        # "[12]" in a summary would later read as a citation marker.
        nodes = nodes_of(
            builder(llm=SummaryLlm(reply="as shown in [12], a summary"), max_levels=1).build(
                leaves("a1", "a2"), lambda: None
            )
        )

        assert "[12]" not in nodes[0].text

    def test_the_gpu_is_released_before_the_llm_is_called(self):
        released = []

        builder().build(leaves("a1", "a2", "b1", "b2"), lambda: released.append(True))

        assert released
