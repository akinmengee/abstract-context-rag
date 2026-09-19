"""Runs against a real Qdrant, because storage bugs only show up in integration.

Start one first:  docker compose up -d qdrant
The tests skip themselves when nothing is listening.
"""

import pytest

from abstractrag.core.config import EmbeddingSettings, QdrantSettings
from abstractrag.database.qdrant_store import QdrantStore
from abstractrag.rag.embedding.bge_m3 import Embedding
from tests.conftest import make_chunk

pytestmark = pytest.mark.integration

DENSE_SIZE = 4


def embedding(dense: list[float], sparse: dict[int, float]) -> Embedding:
    return Embedding(dense=dense, sparse=sparse)


@pytest.fixture
def store() -> QdrantStore:
    store = QdrantStore(
        QdrantSettings(collection="test_documents"),
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
