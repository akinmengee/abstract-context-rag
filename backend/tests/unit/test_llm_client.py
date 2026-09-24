"""LlamaCppClient: payload construction and the max_tokens override.

Verification needs a larger token budget than a normal answer (see
verifier.py) without changing every other call site's budget - these tests
prove the override reaches the request and that omitting it changes nothing.
"""

import httpx

from abstractrag.core.config import LLMSettings
from abstractrag.rag.generation.llm_client import LlamaCppClient


class _FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }


def _client_with_captured_payload(
    monkeypatch, settings: LLMSettings
) -> tuple[LlamaCppClient, dict]:
    captured: dict = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return _FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    return LlamaCppClient(settings), captured


def test_max_tokens_override_reaches_the_request_payload(monkeypatch):
    client, captured = _client_with_captured_payload(monkeypatch, LLMSettings(max_tokens=1024))

    client.complete([{"role": "user", "content": "hi"}], max_tokens=8192)

    assert captured["max_tokens"] == 8192


def test_omitting_the_override_uses_the_configured_default(monkeypatch):
    client, captured = _client_with_captured_payload(monkeypatch, LLMSettings(max_tokens=1024))

    client.complete([{"role": "user", "content": "hi"}])

    assert captured["max_tokens"] == 1024
