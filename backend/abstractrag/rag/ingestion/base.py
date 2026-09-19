"""Ingestion contracts.

Two responsibilities, deliberately separated:
  fetcher  - gets the raw content (arXiv ID -> PDF file, Wikipedia title -> text)
  adapter  - turns raw content into a ParsedDocument

An uploaded PDF skips the fetcher and enters the same adapter, so downloaded and
uploaded documents are parsed by identical code.
"""

from dataclasses import dataclass
from typing import Protocol

from abstractrag.core.errors import InvalidInputError
from abstractrag.rag.models import ParsedDocument


@dataclass(frozen=True)
class SourceInput:
    """Exactly one source must be given: an identifier to fetch, or file bytes to parse."""

    arxiv_id: str | None = None
    wikipedia: str | None = None  # article title or full URL
    file_name: str | None = None
    file_bytes: bytes | None = None

    def __post_init__(self) -> None:
        given = [
            name
            for name, value in (
                ("arxiv_id", self.arxiv_id),
                ("wikipedia", self.wikipedia),
                ("file_bytes", self.file_bytes),
            )
            if value
        ]
        if len(given) != 1:
            raise InvalidInputError(
                f"provide exactly one of arxiv_id, wikipedia, file_bytes (got: {given or 'none'})"
            )
        if self.file_bytes and not self.file_name:
            raise InvalidInputError("file_name is required when uploading file_bytes")

    def describe(self) -> str:
        return self.arxiv_id or self.wikipedia or self.file_name or "<empty>"


class SourceAdapter(Protocol):
    """Turns one source's raw content into the shared ParsedDocument shape."""

    def parse(self, *args, **kwargs) -> ParsedDocument: ...
