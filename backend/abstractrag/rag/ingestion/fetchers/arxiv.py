"""Downloads a paper from arXiv by ID. Only fetching happens here, no parsing."""

import re
from dataclasses import dataclass, field
from pathlib import Path

import arxiv
import httpx

from abstractrag.core.errors import FetchError, SourceNotFoundError
from abstractrag.core.logging import get_logger

logger = get_logger(__name__)

# Accepts "2005.11401", "arXiv:2005.11401v2", "https://arxiv.org/abs/2005.11401".
_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    pdf_path: Path
    authors: list[str] = field(default_factory=list)
    published: str | None = None

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


def normalize_arxiv_id(raw: str) -> str:
    match = _ID_PATTERN.search(raw.strip())
    if not match:
        raise SourceNotFoundError(f"not a valid arXiv ID: {raw!r}")
    return match.group(1)


def fetch_arxiv_paper(raw_id: str, dest_dir: Path) -> ArxivPaper:
    """Download the PDF and return it with the metadata arXiv already knows."""
    arxiv_id = normalize_arxiv_id(raw_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
    except StopIteration as exc:
        raise SourceNotFoundError(f"arXiv has no paper {arxiv_id}") from exc
    except Exception as exc:  # network/API failure
        raise FetchError(f"arXiv lookup failed for {arxiv_id}: {exc}") from exc

    filename = f"{arxiv_id.replace('/', '_')}.pdf"
    target = dest_dir / filename
    if not target.exists():
        # The arxiv package only resolves metadata + pdf_url; fetching the
        # bytes ourselves avoids depending on its (unstable) download helper.
        pdf_url = result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
        logger.info("downloading arXiv %s", arxiv_id)
        try:
            response = httpx.get(pdf_url, follow_redirects=True, timeout=60.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FetchError(f"failed to download {pdf_url}: {exc}") from exc
        target.write_bytes(response.content)

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=result.title.strip(),
        pdf_path=target,
        authors=[author.name for author in result.authors],
        published=result.published.date().isoformat() if result.published else None,
    )
