"""Builds the engine from settings.

One place wires the components together; the API, the CLI and eval scripts all
call get_engine() instead of constructing anything themselves. Models inside the
components load lazily, so importing this module stays cheap.
"""

from functools import lru_cache

from abstractrag.core.config import Settings, get_settings
from abstractrag.database.qdrant_store import QdrantStore
from abstractrag.rag.agents.corrective import CorrectiveAgent
from abstractrag.rag.agents.multi_hop import MultiHopAgent
from abstractrag.rag.chunking.section_aware import SectionAwareChunker
from abstractrag.rag.embedding.bge_m3 import BgeM3Embedder
from abstractrag.rag.engine import RagEngine
from abstractrag.rag.generation.llm_client import LlamaCppClient
from abstractrag.rag.generation.vision import VisionDescriber
from abstractrag.rag.ingestion.resolver import SourceResolver
from abstractrag.rag.raptor.tree import TreeBuilder
from abstractrag.rag.reranking.cross_encoder import CrossEncoderReranker
from abstractrag.rag.retrieval.hybrid import HybridRetriever
from abstractrag.rag.summarization.map_reduce import MapReduceSummarizer
from abstractrag.rag.verification.verifier import ClaimVerifier


def build_agent(settings: Settings, llm: LlamaCppClient) -> CorrectiveAgent | MultiHopAgent | None:
    if settings.agent.mode == "off":
        return None
    threshold = settings.retrieval.score_threshold
    if settings.agent.mode == "corrective":
        return CorrectiveAgent(llm=llm, settings=settings.agent, score_threshold=threshold)
    # Each hop filters: several searches share one context budget.
    hop = CorrectiveAgent(
        llm=llm, settings=settings.agent, score_threshold=threshold, filter_chunks=True
    )
    return MultiHopAgent(llm=llm, settings=settings.agent, corrective=hop)


def build_engine(settings: Settings) -> RagEngine:
    embedder = BgeM3Embedder(settings.embedding)
    store = QdrantStore(settings.qdrant, settings.embedding)
    llm = LlamaCppClient(settings.llm)
    figures_dir = settings.ingestion.resolved_storage_dir() / "figures"
    return RagEngine(
        settings=settings,
        resolver=SourceResolver(settings.ingestion, settings.llm.base_url, figures_dir),
        chunker=SectionAwareChunker(settings.chunking),
        embedder=embedder,
        store=store,
        retriever=HybridRetriever(store, embedder, settings.retrieval),
        reranker=CrossEncoderReranker(settings.reranker),
        llm=llm,
        # The judge is the same local model that answered - no second model to
        # load, and it already has the VRAM.
        verifier=ClaimVerifier(
            llm,
            judge_max_tokens=settings.verification.judge_max_tokens,
            max_prompt_chars=settings.verification.max_prompt_chars,
        ),
        summarizer=MapReduceSummarizer(llm=llm, settings=settings.summarization),
        agent=build_agent(settings, llm),
        tree_builder=TreeBuilder(llm=llm, embedder=embedder, settings=settings.raptor),
        vision=VisionDescriber(settings.vision, settings.llm.base_url)
        if settings.vision.enabled
        else None,
    )


@lru_cache
def get_engine() -> RagEngine:
    return build_engine(get_settings())
