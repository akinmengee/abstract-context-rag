"""Builds the engine from settings.

One place wires the components together; the API, the CLI and eval scripts all
call get_engine() instead of constructing anything themselves. Models inside the
components load lazily, so importing this module stays cheap.
"""

from functools import lru_cache

from abstractrag.core.config import Settings, get_settings
from abstractrag.database.qdrant_store import QdrantStore
from abstractrag.rag.chunking.section_aware import SectionAwareChunker
from abstractrag.rag.embedding.bge_m3 import BgeM3Embedder
from abstractrag.rag.engine import RagEngine
from abstractrag.rag.generation.llm_client import LlamaCppClient
from abstractrag.rag.ingestion.resolver import SourceResolver
from abstractrag.rag.reranking.cross_encoder import CrossEncoderReranker
from abstractrag.rag.retrieval.hybrid import HybridRetriever


def build_engine(settings: Settings) -> RagEngine:
    embedder = BgeM3Embedder(settings.embedding)
    store = QdrantStore(settings.qdrant, settings.embedding)
    return RagEngine(
        settings=settings,
        resolver=SourceResolver(settings.ingestion),
        chunker=SectionAwareChunker(settings.chunking),
        embedder=embedder,
        store=store,
        retriever=HybridRetriever(store, embedder, settings.retrieval),
        reranker=CrossEncoderReranker(settings.reranker),
        llm=LlamaCppClient(settings.llm),
    )


@lru_cache
def get_engine() -> RagEngine:
    return build_engine(get_settings())
