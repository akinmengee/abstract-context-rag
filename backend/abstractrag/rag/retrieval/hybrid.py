"""Stage 1 retrieval: cast a wide net.

Dense finds paraphrases, sparse finds exact terms (model names, metrics, numbers),
and hybrid fuses both with RRF. The mode is a config switch so the three can be
compared on the same golden set.

Fused scores are small by construction (~1/k) and are not comparable to cosine
similarity. Nothing downstream thresholds on them: the abstain decision uses the
reranker score instead.
"""

from abstractrag.core.config import RetrievalSettings
from abstractrag.database.qdrant_store import QdrantStore
from abstractrag.rag.embedding.bge_m3 import BgeM3Embedder
from abstractrag.rag.models import RetrievedChunk
from abstractrag.rag.retrieval import rrf


class HybridRetriever:
    def __init__(
        self,
        store: QdrantStore,
        embedder: BgeM3Embedder,
        settings: RetrievalSettings,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def retrieve(
        self,
        query: str,
        document_id: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        limit = limit or self.settings.candidates
        embedding = self.embedder.embed_one(query)
        mode = self.settings.mode

        if mode == "dense":
            results = self.store.search_dense(embedding, limit, document_id)
            return [RetrievedChunk(chunk=chunk, score=score) for chunk, score in results]

        if mode == "sparse":
            results = self.store.search_sparse(embedding, limit, document_id)
            return [RetrievedChunk(chunk=chunk, score=score) for chunk, score in results]

        dense_results = self.store.search_dense(embedding, limit, document_id)
        sparse_results = self.store.search_sparse(embedding, limit, document_id)
        fused = rrf.fuse([dense_results, sparse_results], k=self.settings.rrf_k)
        return fused[:limit]
