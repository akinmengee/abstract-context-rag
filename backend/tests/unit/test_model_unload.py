"""unload() on the embedder and reranker should only drop+reload weights when
that actually relieves GPU contention with the LLM - a no-op on CPU (see the
docstrings on BgeM3Embedder.unload / CrossEncoderReranker.unload)."""

from abstractrag.core.config import EmbeddingSettings, RerankerSettings
from abstractrag.rag.embedding.bge_m3 import BgeM3Embedder
from abstractrag.rag.reranking.cross_encoder import CrossEncoderReranker


def test_embedder_unload_is_a_noop_on_cpu():
    embedder = BgeM3Embedder(EmbeddingSettings(device="cpu"))
    embedder._model = object()  # simulate an already-loaded model, no real load needed

    embedder.unload()

    assert embedder._model is not None


def test_embedder_unload_drops_the_model_on_cuda():
    embedder = BgeM3Embedder(EmbeddingSettings(device="cuda"))
    embedder._model = object()

    embedder.unload()

    assert embedder._model is None


def test_reranker_unload_is_a_noop_on_cpu():
    reranker = CrossEncoderReranker(RerankerSettings(device="cpu"))
    reranker._model = object()

    reranker.unload()

    assert reranker._model is not None


def test_reranker_unload_drops_the_model_on_cuda():
    reranker = CrossEncoderReranker(RerankerSettings(device="cuda"))
    reranker._model = object()

    reranker.unload()

    assert reranker._model is None
