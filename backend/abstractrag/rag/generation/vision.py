"""Question-conditioned figure descriptions (rag.md 7.9.4).

A fixed, ingest-time caption can never anticipate every question a user
might ask about a figure - "what does Figure 3 show" and "what's the trend
in Figure 3" need different answers, and a single canned caption fits
neither well. This asks the vision model the user's actual question, with
the actual image, at answer time - the same way text retrieval finds a
different passage for a different question, instead of writing one summary
of the whole paper up front.
"""

import base64
from pathlib import Path

import httpx

from abstractrag.core.config import VisionSettings
from abstractrag.core.logging import get_logger

logger = get_logger(__name__)


class VisionDescriber:
    def __init__(self, settings: VisionSettings, base_url: str) -> None:
        self.settings = settings
        self.base_url = base_url

    def describe(self, image_path: Path, question: str) -> str:
        """The vision model's answer about one figure; "" on any failure.

        A missing file or a failed call should not break the whole answer -
        the caller falls back to whatever text the chunk already had.
        """
        try:
            image_b64 = base64.b64encode(image_path.read_bytes()).decode()
        except OSError as exc:
            logger.warning("could not read figure image %s: %s", image_path, exc)
            return ""

        payload = {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            logger.warning("vision call failed for %s: %s", image_path, exc)
            return ""
