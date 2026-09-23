"""Qdrant collection schema and access.

One collection holds both vectors bge-m3 produces: a named dense vector and a
named sparse vector. They are queried separately and fused in the retrieval layer,
which keeps each retrieval mode (dense / sparse / hybrid) measurable on its own.
"""

from typing import Any

from qdrant_client import QdrantClient, models

from abstractrag.core.config import EmbeddingSettings, QdrantSettings
from abstractrag.core.logging import get_logger
from abstractrag.rag.embedding.bge_m3 import Embedding
from abstractrag.rag.models import Chunk

logger = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
_SCROLL_PAGE = 256


class QdrantStore:
    def __init__(self, settings: QdrantSettings, embedding: EmbeddingSettings) -> None:
        self.settings = settings
        self.embedding = embedding
        self.client = QdrantClient(url=settings.url)

    @property
    def collection(self) -> str:
        return self.settings.collection

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return

        logger.info("creating collection %s", self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=self.embedding.dense_size, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )
        # Needed to delete or filter a single document quickly.
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[Embedding]) -> None:
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    DENSE_VECTOR: embedding.dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=list(embedding.sparse.keys()),
                        values=list(embedding.sparse.values()),
                    ),
                },
                payload={
                    "document_id": chunk.document_id,
                    "chunk": chunk.model_dump(mode="json"),
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def delete_document(self, document_id: str) -> None:
        """Re-ingesting a source replaces it; chunk IDs are stable but the count may shrink."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=_document_filter(document_id)),
        )

    def search_dense(
        self, embedding: Embedding, limit: int, document_id: str | None = None
    ) -> list[tuple[Chunk, float]]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=embedding.dense,
            using=DENSE_VECTOR,
            limit=limit,
            with_payload=True,
            query_filter=_document_filter(document_id) if document_id else None,
        )
        return _to_chunks(response.points)

    def search_sparse(
        self, embedding: Embedding, limit: int, document_id: str | None = None
    ) -> list[tuple[Chunk, float]]:
        if not embedding.sparse:
            return []

        response = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(
                indices=list(embedding.sparse.keys()),
                values=list(embedding.sparse.values()),
            ),
            using=SPARSE_VECTOR,
            limit=limit,
            with_payload=True,
            query_filter=_document_filter(document_id) if document_id else None,
        )
        return _to_chunks(response.points)

    def list_chunks(self, document_id: str) -> list[Chunk]:
        """Every chunk of one document, in reading order.

        Summarising is a global task, so it needs the whole document rather than
        what a query retrieves - this scrolls the collection instead of searching
        it, paging until Qdrant stops handing back an offset.
        """
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=_document_filter(document_id),
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
            )
            chunks.extend(Chunk.model_validate(point.payload["chunk"]) for point in points)
            if offset is None:
                return sorted(chunks, key=lambda chunk: chunk.index)

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        """One entry per ingested document, built from the chunk payloads."""
        points, _ = self.client.scroll(
            collection_name=self.collection, limit=limit * 50, with_payload=True
        )
        documents: dict[str, dict[str, Any]] = {}
        for point in points:
            chunk = (point.payload or {}).get("chunk", {})
            metadata = chunk.get("metadata", {})
            entry = documents.setdefault(
                chunk.get("document_id", ""),
                {
                    "document_id": chunk.get("document_id", ""),
                    "title": metadata.get("title", ""),
                    "source_type": metadata.get("source_type", ""),
                    "origin": metadata.get("origin", ""),
                    "chunk_count": 0,
                },
            )
            entry["chunk_count"] += 1
        return list(documents.values())[:limit]


def _document_filter(document_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )


def _to_chunks(points: list[Any]) -> list[tuple[Chunk, float]]:
    return [(Chunk.model_validate(point.payload["chunk"]), point.score) for point in points]
