"""Request models for the API. Responses reuse rag/models.py directly instead
of being duplicated here, except DocumentSummary which trims the fields the
document list endpoint actually needs."""

from pydantic import BaseModel, Field, model_validator

from abstractrag.rag.ingestion.base import SourceInput


class IngestRequest(BaseModel):
    """Ingest by identifier. File uploads use the multipart endpoint instead."""

    arxiv_id: str | None = Field(default=None, examples=["2005.11401"])
    wikipedia: str | None = Field(default=None, examples=["Retrieval-augmented generation"])

    @model_validator(mode="after")
    def exactly_one_source(self) -> "IngestRequest":
        if bool(self.arxiv_id) == bool(self.wikipedia):
            raise ValueError("provide exactly one of arxiv_id or wikipedia")
        return self

    def to_source_input(self) -> SourceInput:
        return SourceInput(arxiv_id=self.arxiv_id, wikipedia=self.wikipedia)


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    source_type: str
    origin: str
    chunk_count: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, examples=["What datasets does the paper evaluate on?"])
    # Optional: scope the answer to one document instead of the whole collection.
    document_id: str | None = None
