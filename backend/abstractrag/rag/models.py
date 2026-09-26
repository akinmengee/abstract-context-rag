"""Domain models shared by every stage of the pipeline.

These are the contract between layers: source adapters produce a ParsedDocument,
chunking turns it into Chunks, retrieval returns RetrievedChunks, generation
returns an Answer. Nothing below ingestion knows which source a chunk came from.
"""

import uuid
from enum import Enum

from pydantic import BaseModel, Field

# Stable namespace so re-ingesting the same source yields the same IDs.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def document_id_for(origin: str) -> str:
    """Deterministic document ID, so re-ingesting a source updates instead of duplicating."""
    return str(uuid.uuid5(_ID_NAMESPACE, origin))


def chunk_id_for(document_id: str, index: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{document_id}:{index}"))


def node_id_for(document_id: str, level: int, position: int) -> str:
    """Deterministic ID for a RAPTOR tree node, so rebuilding a tree replaces it."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{document_id}:tree:{level}:{position}"))


class SourceType(str, Enum):
    PDF = "pdf"
    WIKIPEDIA = "wikipedia"


class BlockType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CAPTION = "caption"
    FORMULA = "formula"
    FIGURE = "figure"
    OTHER = "other"


class DocumentBlock(BaseModel):
    """One layout block of a parsed document, in reading order."""

    text: str
    block_type: BlockType = BlockType.PARAGRAPH
    section_path: list[str] = Field(default_factory=list)  # heading hierarchy, outermost first
    page: int | None = None
    # A FIGURE block's saved image, for query-time vision (rag.md 7.9.4) -
    # None for every other block type, and for a figure when
    # ingestion.figures is disabled.
    image_path: str | None = None


class ParsedDocument(BaseModel):
    """Source-agnostic parser output. Every source adapter returns this shape."""

    document_id: str
    title: str
    source_type: SourceType
    origin: str  # arXiv ID, Wikipedia URL, or original filename
    blocks: list[DocumentBlock]
    extra: dict[str, str] = Field(default_factory=dict)  # authors, revision_id, ...


class ChunkMetadata(BaseModel):
    """Everything needed to cite a chunk back to its place in the source."""

    document_id: str
    title: str
    source_type: SourceType
    origin: str
    section: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None
    block_types: list[BlockType] = Field(default_factory=list)
    # Saved figure images this chunk's blocks carry (rag.md 7.9.4) - usually
    # 0 or 1, more if several figures land in one chunk. Never sent to Qdrant
    # for embedding, only read back at answer time.
    image_paths: list[str] = Field(default_factory=list)
    # 0 for a chunk of the source; n > 0 for a RAPTOR summary node n levels up.
    level: int = 0
    # Tree nodes only: the leaf chunks the summary covers. Verification checks a
    # claim citing a node against these leaves' text, never the summary itself.
    source_ids: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    index: int
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    """A chunk plus the score of whichever stage produced it."""

    chunk: Chunk
    score: float
    # Set once the cross-encoder has run; kept separate so ablations can compare stages.
    rerank_score: float | None = None

    @property
    def effective_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


class SectionSummary(BaseModel):
    """One map step's output: what a [n] marker in a summary points at.

    `chunk_ids` is the section's full source text, which is what verification
    checks a summary claim against - never this summary of it, or a claim
    invented here would confirm itself.
    """

    marker: int
    section: str
    text: str
    chunk_ids: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """One numbered context block the answer is allowed to point at."""

    marker: int  # the [n] used in the answer text
    chunk_id: str
    title: str
    section: str | None = None
    page: int | None = None
    origin: str


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"  # the cited passage backs this claim
    UNSUPPORTED = "unsupported"  # it cites something, but that passage does not say this
    UNCITED = "uncited"  # asserted with no source at all


class VerifiedClaim(BaseModel):
    """One sentence of an answer, checked against the passage it cited."""

    text: str
    markers: list[int] = Field(default_factory=list)
    verdict: ClaimVerdict
    # Only failures need explaining, so this stays empty for supported claims.
    reason: str = ""


class AgentStep(BaseModel):
    """One search an agent ran: what it searched for and what survived grading."""

    query: str
    retrieved: int
    kept: int


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    # True when retrieval was too weak or the model reported the source has no answer.
    abstained: bool = False
    used_chunks: list[RetrievedChunk] = Field(default_factory=list)
    # Empty when verification is disabled, or when the answer abstained and so
    # claimed nothing to check.
    verified_claims: list[VerifiedClaim] = Field(default_factory=list)
    # Filled only by map-reduce summaries, where a citation marker means a whole
    # section rather than one retrieved chunk (`used_chunks` stays empty there,
    # because no retrieval ran). See rag.md 8.1.
    section_summaries: list[SectionSummary] = Field(default_factory=list)
    # Filled only when an agent chose the context (agent.mode != off): the
    # searches it ran, which is what explains a multi-hop answer or a miss.
    agent_steps: list[AgentStep] = Field(default_factory=list)


class IngestResult(BaseModel):
    document_id: str
    title: str
    source_type: SourceType
    origin: str
    chunk_count: int
    page_count: int | None = None
    # RAPTOR summary nodes built at ingest; 0 unless raptor.build_on_ingest.
    tree_nodes: int = 0
