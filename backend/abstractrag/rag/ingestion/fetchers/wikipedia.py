"""Fetches a single Wikipedia article as plain text.

Uses the TextExtracts API (`explaintext`) instead of the HTML endpoint: it returns
clean text with `== Heading ==` markers, which is exactly what heading-based
chunking needs and avoids an HTML parsing dependency.
Known limitation: infoboxes and tables are dropped by this endpoint.
"""

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import httpx

from abstractrag.core.errors import FetchError, SourceNotFoundError

API_URL = "https://en.wikipedia.org/w/api.php"


@dataclass
class WikipediaArticle:
    title: str
    url: str
    revision_id: str
    text: str


def normalize_article_title(raw: str) -> str:
    """Accepts a plain title or a full wikipedia.org URL."""
    value = raw.strip()
    if value.startswith("http"):
        path = urlparse(value).path
        match = re.match(r"^/wiki/(?P<title>.+)$", path)
        if not match:
            raise SourceNotFoundError(f"not a Wikipedia article URL: {raw!r}")
        value = unquote(match.group("title"))
    return value.replace("_", " ")


def fetch_wikipedia_article(raw: str, user_agent: str) -> WikipediaArticle:
    title = normalize_article_title(raw)
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info|revisions",
        "explaintext": "1",
        "redirects": "1",
        "inprop": "url",
        "rvprop": "ids",
        "titles": title,
    }

    try:
        response = httpx.get(
            API_URL, params=params, headers={"User-Agent": user_agent}, timeout=30.0
        )
        response.raise_for_status()
        pages = response.json()["query"]["pages"]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise FetchError(f"Wikipedia request failed for {title!r}: {exc}") from exc

    page = next(iter(pages.values()))
    if "missing" in page or not page.get("extract"):
        raise SourceNotFoundError(f"Wikipedia has no article {title!r}")

    return WikipediaArticle(
        title=page["title"],
        url=page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
        revision_id=str(page.get("revisions", [{}])[0].get("revid", "")),
        text=page["extract"],
    )
