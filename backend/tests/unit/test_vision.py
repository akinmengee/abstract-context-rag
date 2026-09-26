"""VisionDescriber: sends the real question with the image, not a fixed prompt.

A canned "describe this figure" caption cannot anticipate every question a
user might ask; this proves the actual question text reaches the request
(rag.md 7.9.4), and that a missing image or a failed call degrades to an
empty string rather than raising - a bad figure lookup should not break the
whole answer.
"""

import httpx

from abstractrag.core.config import VisionSettings
from abstractrag.rag.generation.vision import VisionDescriber


class _FakeResponse:
    status_code = 200

    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self.content}}]}


def test_sends_the_actual_question_not_a_generic_prompt(tmp_path, monkeypatch):
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"not a real png, just bytes for the test")
    captured: dict = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return _FakeResponse("a bar chart of accuracy rising with more documents")

    monkeypatch.setattr(httpx, "post", fake_post)
    describer = VisionDescriber(
        VisionSettings(model="granite3.2-vision:2b"), "http://127.0.0.1:11434/v1"
    )

    result = describer.describe(image_path, "What does Figure 3 show?")

    assert result == "a bar chart of accuracy rising with more documents"
    assert captured["model"] == "granite3.2-vision:2b"
    text_part = captured["messages"][0]["content"][0]
    assert text_part == {"type": "text", "text": "What does Figure 3 show?"}
    image_part = captured["messages"][0]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_returns_empty_string_when_the_image_file_is_missing(tmp_path):
    describer = VisionDescriber(VisionSettings(), "http://127.0.0.1:11434/v1")

    result = describer.describe(tmp_path / "missing.png", "What is this?")

    assert result == ""


def test_returns_empty_string_on_a_failed_call(tmp_path, monkeypatch):
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"bytes")

    def fake_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    describer = VisionDescriber(VisionSettings(), "http://127.0.0.1:11434/v1")

    result = describer.describe(image_path, "What is this?")

    assert result == ""
