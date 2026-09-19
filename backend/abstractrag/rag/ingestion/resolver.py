"""Turns a SourceInput into a ParsedDocument by picking the right fetcher + adapter."""

import re
from pathlib import Path

from abstractrag.core.config import IngestionSettings
from abstractrag.core.errors import InvalidInputError
from abstractrag.rag.ingestion.base import SourceInput
from abstractrag.rag.ingestion.fetchers.arxiv import fetch_arxiv_paper
from abstractrag.rag.ingestion.fetchers.wikipedia import fetch_wikipedia_article
from abstractrag.rag.ingestion.sources.pdf_source import PdfSource
from abstractrag.rag.ingestion.sources.wikipedia_source import WikipediaSource
from abstractrag.rag.models import ParsedDocument

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


class SourceResolver:
    def __init__(self, settings: IngestionSettings) -> None:
        self.settings = settings
        self.pdf_source = PdfSource()
        self.wikipedia_source = WikipediaSource()

    @property
    def storage_dir(self) -> Path:
        return self.settings.resolved_storage_dir()

    def resolve(self, source: SourceInput) -> ParsedDocument:
        if source.arxiv_id:
            paper = fetch_arxiv_paper(source.arxiv_id, self.storage_dir / "arxiv")
            return self.pdf_source.parse(
                paper.pdf_path,
                origin=paper.abs_url,
                title=paper.title,
                extra={
                    "arxiv_id": paper.arxiv_id,
                    "authors": ", ".join(paper.authors),
                    "published": paper.published or "",
                },
            )

        if source.wikipedia:
            article = fetch_wikipedia_article(source.wikipedia, self.settings.user_agent)
            return self.wikipedia_source.parse(article)

        return self._parse_upload(source)

    def _parse_upload(self, source: SourceInput) -> ParsedDocument:
        assert source.file_bytes and source.file_name  # guaranteed by SourceInput
        if not source.file_name.lower().endswith(".pdf"):
            raise InvalidInputError("only PDF uploads are supported")

        # Keep the basename only: an uploaded name must never escape the storage dir.
        safe_name = _UNSAFE_CHARS.sub("_", Path(source.file_name).name)
        upload_dir = self.storage_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = upload_dir / safe_name
        pdf_path.write_bytes(source.file_bytes)

        return self.pdf_source.parse(pdf_path, origin=f"upload:{safe_name}")
