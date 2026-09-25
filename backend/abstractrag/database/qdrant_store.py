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
                    "level": chunk.metadata.level,
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
            points_selector=models.FilterSelector(filter=_scope(document_id)),
        )

    def delete_tree(self, document_id: str) -> None:
        """Remove one document's summary nodes; its leaves stay."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[_document_condition(document_id), _TREE_NODE])
            ),
        )

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Chunks by ID, in the order asked; unknown IDs are skipped."""
        if not chunk_ids:
            return []
        points = self.client.retrieve(
            collection_name=self.collection, ids=chunk_ids, with_payload=True
        )
        by_id = {str(point.id): Chunk.model_validate(point.payload["chunk"]) for point in points}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def search_dense(
        self,
        embedding: Embedding,
        limit: int,
        document_id: str | None = None,
        include_tree: bool = False,
    ) -> list[tuple[Chunk, float]]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=embedding.dense,
            using=DENSE_VECTOR,
            limit=limit,
            with_payload=True,
            query_filter=_scope(document_id, leaves_only=not include_tree),
        )
        return _to_chunks(response.points)

    def search_sparse(
        self,
        embedding: Embedding,
        limit: int,
        document_id: str | None = None,
        include_tree: bool = False,
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
            query_filter=_scope(document_id, leaves_only=not include_tree),
        )
        return _to_chunks(response.points)

    def list_chunks(self, document_id: str, level: int = 0) -> list[Chunk]:
        """Every chunk of one document at one tree level, in reading order.

        Level 0 is the document itself; summarising and tree building need all
        of it rather than what a query retrieves, so this scrolls instead of
        searching, paging until Qdrant stops handing back an offset.
        """
        if level == 0:
            scroll_filter = _scope(document_id, leaves_only=True)
        else:
            scroll_filter = _scope(document_id, level=level)
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
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
            if (point.payload or {}).get("level", 0):
                continue  # a tree node, not part of the document's own chunks
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


# Points written before tree levels existed have no "level" key; a range
# condition never matches a missing key, so must_not keeps them as leaves.
_TREE_NODE = models.FieldCondition(key="level", range=models.Range(gt=0))


def _document_condition(document_id: str) -> models.FieldCondition:
    return models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))


def _scope(
    document_id: str | None, leaves_only: bool = False, level: int | None = None
) -> models.Filter | None:
    must: list[models.Condition] = []
    if document_id:
        must.append(_document_condition(document_id))
    if level:
        must.append(models.FieldCondition(key="level", match=models.MatchValue(value=level)))
    must_not = [_TREE_NODE] if leaves_only else []
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def _to_chunks(points: list[Any]) -> list[tuple[Chunk, float]]:
    return [(Chunk.model_validate(point.payload["chunk"]), point.score) for point in points]
