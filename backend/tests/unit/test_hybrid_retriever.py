"""HybridRetriever: which store searches it runs, and with what scope."""

import pytest

from abstractrag.core.config import RetrievalSettings
from abstractrag.rag.embedding.bge_m3 import Embedding
from abstractrag.rag.retrieval.hybrid import HybridRetriever
from tests.conftest import make_chunk


class RecordingStore:
    """Returns one chunk per search and records the include_tree flag it was given."""

    def __init__(self) -> None:
        self.include_tree: list[bool] = []

    def search_dense(self, embedding, limit, document_id=None, include_tree=False):
        self.include_tree.append(include_tree)
        return [(make_chunk("dense hit", index=0), 0.9)]

    def search_sparse(self, embedding, limit, document_id=None, include_tree=False):
        self.include_tree.append(include_tree)
        return [(make_chunk("sparse hit", index=1), 0.8)]


class StubEmbedder:
    def embed_one(self, text: str) -> Embedding:
        return Embedding(dense=[1.0, 0.0], sparse={1: 1.0})


@pytest.mark.parametrize("mode, searches", [("dense", 1), ("sparse", 1), ("hybrid", 2)])
def test_tree_nodes_stay_hidden_by_default(mode, searches):
    store = RecordingStore()

    HybridRetriever(store, StubEmbedder(), RetrievalSettings(mode=mode)).retrieve("q")

    assert store.include_tree == [False] * searches


@pytest.mark.parametrize("mode, searches", [("dense", 1), ("sparse", 1), ("hybrid", 2)])
def test_tree_nodes_are_requested_when_configured(mode, searches):
    store = RecordingStore()
    settings = RetrievalSettings(mode=mode, include_tree_nodes=True)

    HybridRetriever(store, StubEmbedder(), settings).retrieve("q")

    assert store.include_tree == [True] * searches
