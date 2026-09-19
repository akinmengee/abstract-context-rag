"""Client for the llama.cpp server (OpenAI-compatible /chat/completions).

Kept deliberately thin: swapping llama.cpp for any other OpenAI-compatible server
is a config change, not a code change.
"""

import json
from collections.abc import Iterator

import httpx

from abstractrag.core.config import LLMSettings
from abstractrag.core.errors import LLMError


class LlamaCppClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = self._payload(messages, stream=False)
        try:
            response = httpx.post(
                f"{self.settings.base_url}/chat/completions",
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPError as exc:
            raise LLMError(f"llama.cpp request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"unexpected response from llama.cpp: {exc}") from exc

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Yield content deltas as they arrive."""
        payload = self._payload(messages, stream=True)
        try:
            with httpx.stream(
                "POST",
                f"{self.settings.base_url}/chat/completions",
                json=payload,
                timeout=self.settings.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    delta = _content_delta(line)
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise LLMError(f"llama.cpp stream failed: {exc}") from exc

    def health(self) -> bool:
        try:
            response = httpx.get(f"{self.settings.base_url}/models", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _payload(self, messages: list[dict[str, str]], stream: bool) -> dict:
        return {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": stream,
        }


def _content_delta(line: str) -> str | None:
    """Parse one SSE line of an OpenAI-style stream."""
    if not line.startswith("data: "):
        return None
    data = line.removeprefix("data: ").strip()
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data)["choices"][0]["delta"].get("content")
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
