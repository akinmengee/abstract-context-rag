"""Runs against a real Qdrant, because storage bugs only show up in integration.

Start one first:  docker compose up -d qdrant
The tests skip themselves when nothing is listening.
"""

import contextlib
import uuid

import pytest

from abstractrag.core.config import EmbeddingSettings, QdrantSettings
from abstractrag.database.qdrant_store import QdrantStore
from abstractrag.rag.embedding.bge_m3 import Embedding
from abstractrag.rag.models import node_id_for
from tests.conftest import make_chunk

pytestmark = pytest.mark.integration

DENSE_SIZE = 4


def embedding(dense: list[float], sparse: dict[int, float]) -> Embedding:
    return Embedding(dense=dense, sparse=sparse)


@pytest.fixture
def store() -> QdrantStore:
    # A fresh name per test: with the storage on a synced folder (OneDrive), a
    # deleted collection's directory sometimes lingers and blocks re-creating
    # the same name ("Collection data already exists").
    store = QdrantStore(
        QdrantSettings(collection=f"test_documents_{uuid.uuid4().hex[:8]}"),
        EmbeddingSettings(dense_size=DENSE_SIZE),
    )
    try:
        store.client.get_collections()
    except Exception:  # noqa: BLE001 - any connection problem means "not available"
        pytest.skip("Qdrant is not running")

    if store.client.collection_exists(store.collection):
        store.client.delete_collection(store.collection)
    store.ensure_collection()
    yield store
    # Cleanup, not the test: a failed delete on the synced folder must not fail it.
    with contextlib.suppress(Exception):
        store.client.delete_collection(store.collection)


def test_dense_search_finds_the_closest_chunk(store: QdrantStore):
    near = make_chunk("the answer", index=0)
    far = make_chunk("unrelated", index=1)
    store.upsert_chunks(
        [near, far],
        [embedding([1.0, 0.0, 0.0, 0.0], {}), embedding([0.0, 1.0, 0.0, 0.0], {})],
    )

    results = store.search_dense(embedding([0.9, 0.1, 0.0, 0.0], {}), limit=2)

    assert results[0][0].chunk_id == near.chunk_id


def test_sparse_search_matches_shared_terms(store: QdrantStore):
    chunk = make_chunk("keyword chunk", index=0)
    store.upsert_chunks([chunk], [embedding([1.0, 0.0, 0.0, 0.0], {7: 0.8, 9: 0.5})])

    results = store.search_sparse(embedding([0.0] * DENSE_SIZE, {7: 1.0}), limit=5)

    assert [found.chunk_id for found, _ in results] == [chunk.chunk_id]


def test_reingesting_a_document_replaces_it(store: QdrantStore):
    chunk = make_chunk("first version", index=0)
    store.upsert_chunks([chunk], [embedding([1.0, 0.0, 0.0, 0.0], {})])

    store.delete_document(chunk.document_id)

    assert store.search_dense(embedding([1.0, 0.0, 0.0, 0.0], {}), limit=5) == []


def test_documents_are_listed_with_their_chunk_counts(store: QdrantStore):
    chunks = [make_chunk(f"chunk {index}", index=index) for index in range(3)]
    store.upsert_chunks(chunks, [embedding([1.0, 0.0, 0.0, 0.0], {}) for _ in chunks])

    documents = store.list_documents()

    assert len(documents) == 1
    assert documents[0]["chunk_count"] == 3
    assert documents[0]["title"] == "Test Paper"


def tree_node(leaves: list, level: int = 1, position: int = 0):
    """A summary node over some leaves, built the way the tree builder builds it."""
    first = leaves[0]
    return first.model_copy(
        update={
            "chunk_id": node_id_for(first.document_id, level, position),
            "index": position,
            "text": "summary of the leaves",
            "metadata": first.metadata.model_copy(
                update={"level": level, "source_ids": [leaf.chunk_id for leaf in leaves]}
            ),
        }
    )


VECTOR = [1.0, 0.0, 0.0, 0.0]


def test_tree_nodes_are_hidden_from_search_unless_asked_for(store: QdrantStore):
    leaf = make_chunk("leaf", index=0)
    node = tree_node([leaf])
    store.upsert_chunks([leaf, node], [embedding(VECTOR, {3: 1.0}), embedding(VECTOR, {3: 1.0})])

    default = store.search_dense(embedding(VECTOR, {}), limit=5)
    with_tree = store.search_dense(embedding(VECTOR, {}), limit=5, include_tree=True)
    sparse = store.search_sparse(embedding(VECTOR, {3: 1.0}), limit=5)

    assert [chunk.chunk_id for chunk, _ in default] == [leaf.chunk_id]
    assert {chunk.chunk_id for chunk, _ in with_tree} == {leaf.chunk_id, node.chunk_id}
    assert [chunk.chunk_id for chunk, _ in sparse] == [leaf.chunk_id]


def test_list_chunks_returns_leaves_by_default_and_one_level_on_request(store: QdrantStore):
    leaves = [make_chunk(f"leaf {i}", index=i) for i in range(2)]
    node = tree_node(leaves)
    store.upsert_chunks([*leaves, node], [embedding(VECTOR, {}) for _ in range(3)])

    assert [c.chunk_id for c in store.list_chunks(node.document_id)] == [
        leaf.chunk_id for leaf in leaves
    ]
    assert [c.chunk_id for c in store.list_chunks(node.document_id, level=1)] == [node.chunk_id]


def test_delete_tree_keeps_the_leaves(store: QdrantStore):
    leaf = make_chunk("leaf", index=0)
    store.upsert_chunks([leaf, tree_node([leaf])], [embedding(VECTOR, {}), embedding(VECTOR, {})])

    store.delete_tree(leaf.document_id)

    assert store.list_chunks(leaf.document_id, level=1) == []
    assert [c.chunk_id for c in store.list_chunks(leaf.document_id)] == [leaf.chunk_id]


def test_get_chunks_returns_them_in_the_order_asked(store: QdrantStore):
    chunks = [make_chunk(f"c{i}", index=i) for i in range(3)]
    store.upsert_chunks(chunks, [embedding(VECTOR, {}) for _ in chunks])

    found = store.get_chunks([chunks[2].chunk_id, chunks[0].chunk_id])

    assert [c.text for c in found] == ["c2", "c0"]


def test_document_listing_counts_leaves_not_tree_nodes(store: QdrantStore):
    leaves = [make_chunk(f"leaf {i}", index=i) for i in range(2)]
    store.upsert_chunks([*leaves, tree_node(leaves)], [embedding(VECTOR, {}) for _ in range(3)])

    assert store.list_documents()[0]["chunk_count"] == 2


def test_points_stored_before_tree_levels_existed_count_as_leaves(store: QdrantStore):
    # Every point ingested before this change has no "level" in its payload.
    leaf = make_chunk("old leaf", index=0)
    store.upsert_chunks([leaf], [embedding(VECTOR, {})])
    store.client.delete_payload(
        collection_name=store.collection, keys=["level"], points=[leaf.chunk_id]
    )

    assert [c.chunk_id for c in store.list_chunks(leaf.document_id)] == [leaf.chunk_id]
    assert [c.chunk_id for c, _ in store.search_dense(embedding(VECTOR, {}), limit=5)] == [
        leaf.chunk_id
    ]
