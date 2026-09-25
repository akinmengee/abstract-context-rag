"""Building a document's RAPTOR tree, bottom-up.

Leaves are the document's chunks. Each level clusters the level below by
embedding, summarises every cluster with the map-reduce map prompt, and embeds
the summaries for the next level. The cost is one LLM call per cluster, paid
once per document at indexing time instead of on every global question.
"""

from collections.abc import Callable
from dataclasses import dataclass

from abstractrag.core.config import RaptorSettings
from abstractrag.core.errors import LLMError
from abstractrag.core.logging import get_logger
from abstractrag.rag.chunking.section_aware import embedding_text
from abstractrag.rag.embedding.bge_m3 import BgeM3Embedder, Embedding
from abstractrag.rag.generation.llm_client import LlamaCppClient
from abstractrag.rag.generation.prompts import CITATION_MARKER
from abstractrag.rag.models import Chunk, node_id_for
from abstractrag.rag.raptor.clustering import cluster
from abstractrag.rag.summarization import prompts
from abstractrag.rag.summarization.sections import SectionGroup

logger = get_logger(__name__)

_LABEL_SECTIONS = 3


@dataclass
class TreeBuilder:
    llm: LlamaCppClient
    embedder: BgeM3Embedder
    settings: RaptorSettings

    def build(
        self, leaves: list[Chunk], release_gpu: Callable[[], None]
    ) -> list[tuple[Chunk, Embedding]]:
        """Summary nodes for one document's leaves, level by level; leaves are not returned."""
        built: list[tuple[Chunk, Embedding]] = []
        below = leaves
        vectors = [e.dense for e in self.embedder.embed([embedding_text(c) for c in leaves])]

        for level in range(1, self.settings.max_levels + 1):
            groups = cluster(vectors, self.settings.cluster_size)
            if len(groups) >= len(below):
                break  # nothing left to merge
            release_gpu()
            nodes = [
                node
                for position, group in enumerate(groups)
                if (node := self._node([below[i] for i in group], level, position)) is not None
            ]
            if not nodes:
                break
            embeddings = self.embedder.embed([embedding_text(node) for node in nodes])
            built.extend(zip(nodes, embeddings, strict=True))
            below, vectors = nodes, [e.dense for e in embeddings]
        return built

    def _node(self, members: list[Chunk], level: int, position: int) -> Chunk | None:
        label = _label(members)
        logger.info("tree level %d, cluster %d: %s", level, position, label)
        try:
            reply = self.llm.complete(
                prompts.build_map_messages(
                    prompts.DEFAULT_REQUEST, SectionGroup(section=label, chunks=members)
                )
            )
        except LLMError as exc:
            logger.warning("skipping cluster %r after LLM failure: %s", label, exc)
            return None
        text = CITATION_MARKER.sub("", reply).strip()
        if not text or prompts.NOTHING_RELEVANT in text:
            return None

        first = members[0]
        pages = [m.metadata.page for m in members if m.metadata.page is not None]
        return Chunk(
            chunk_id=node_id_for(first.document_id, level, position),
            document_id=first.document_id,
            index=position,
            text=text,
            metadata=first.metadata.model_copy(
                update={
                    "section": label,
                    "section_path": [label],
                    "page": min(pages) if pages else None,
                    "block_types": [],
                    "level": level,
                    "source_ids": _leaf_ids(members),
                }
            ),
        )


def _leaf_ids(members: list[Chunk]) -> list[str]:
    """The source chunks under these members, flattened through any lower nodes."""
    ids: list[str] = []
    for member in members:
        for leaf in member.metadata.source_ids if member.metadata.level else [member.chunk_id]:
            if leaf not in ids:
                ids.append(leaf)
    return ids


def _label(members: list[Chunk]) -> str:
    """The sections a cluster covers, in order: what its citations will show."""
    sections: list[str] = []
    for member in members:
        name = member.metadata.section or member.metadata.title
        if name not in sections:
            sections.append(name)
    shown = "; ".join(sections[:_LABEL_SECTIONS])
    extra = len(sections) - _LABEL_SECTIONS
    return f"{shown} (+{extra} more)" if extra > 0 else shown
