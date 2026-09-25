"""Client for the llama.cpp server (OpenAI-compatible /chat/completions).

Kept deliberately thin: swapping llama.cpp for any other OpenAI-compatible server
is a config change, not a code change.
"""

import json
from collections.abc import Iterator

import httpx

from abstractrag.core.config import LLMSettings
from abstractrag.core.errors import LLMError
from abstractrag.core.logging import get_logger

logger = get_logger(__name__)


class LlamaCppClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload = self._payload(messages, stream=False, max_tokens=max_tokens)
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            response = httpx.post(
                f"{self.settings.base_url}/chat/completions",
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            self._warn_if_truncated(body.get("usage") or {})
            text = body["choices"][0]["message"]["content"].strip()
            if not text:
                # Most likely cause: max_tokens ran out during Qwen3's "thinking"
                # pass before any content was written - this makes that visible
                # instead of a silent empty answer.
                choice = body["choices"][0]
                logger.warning(
                    "empty content: finish_reason=%s usage=%s reasoning_chars=%d",
                    choice.get("finish_reason"),
                    body.get("usage"),
                    len(choice.get("message", {}).get("reasoning") or ""),
                )
            return text
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

    def _warn_if_truncated(self, usage: dict) -> None:
        """Ollama cuts a prompt longer than its context window from the front,
        without an error - the system prompt goes first. A prompt that exactly
        fills the window is the only visible trace of that."""
        prompt_tokens = usage.get("prompt_tokens") or 0
        if prompt_tokens >= self.settings.ctx_size:
            logger.warning(
                "prompt filled the whole context window (%d tokens, ctx_size=%d): "
                "it was probably truncated and the system prompt lost",
                prompt_tokens,
                self.settings.ctx_size,
            )

    def _payload(
        self, messages: list[dict[str, str]], stream: bool, max_tokens: int | None = None
    ) -> dict:
        return {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
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
