"""HTTP-layer tests for conversation CRUD and the conversation-scoped /chat
endpoints: per-user isolation, message persistence, and history threading."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from abstractrag.accounts import models  # noqa: F401 - registers tables on SQLModel.metadata
from abstractrag.api.dependencies import get_session
from abstractrag.core.container import get_engine
from abstractrag.core.db import get_db_engine
from abstractrag.main import app
from abstractrag.rag.models import SectionSummary
from tests.conftest import make_chunk
from tests.unit.test_engine import FakeStore, FakeSummarizer, build_engine, candidate


def make_client_with_fake_engine(response_text: str = "Grounded answer [1].", **engine_kwargs):
    # Same in-memory engine backs both overrides: SessionDep (request-scoped
    # reads/writes) and DbEngineDep (api/chat.py's streaming persistence,
    # which opens its own Session rather than reusing SessionDep - see that
    # module for why). Without overriding DbEngineDep too, streaming
    # persistence would silently write through to the real get_db_engine()
    # instead of this test's isolated database.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)

    def override_session():
        with Session(engine) as session:
            yield session

    fake_engine = build_engine(
        [candidate("relevant passage", 0)], [0.9], response_text, **engine_kwargs
    )
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_db_engine] = lambda: engine
    app.dependency_overrides[get_engine] = lambda: fake_engine
    return TestClient(app), fake_engine


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def register_and_login(
    client: TestClient, email: str = "a@example.com", password: str = "hunter2222"
) -> str:
    client.post("/api/v1/auth/register", json={"email": email, "password": password})
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    return response.json()["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_a_created_conversation_is_scoped_to_its_creator():
    client, _ = make_client_with_fake_engine()
    token = register_and_login(client)

    response = client.post(
        "/api/v1/conversations",
        json={"document_id": "doc-1", "title": "My paper"},
        headers=auth_headers(token),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "My paper"
    assert body["document_id"] == "doc-1"

    listing = client.get("/api/v1/conversations", headers=auth_headers(token))
    assert [c["id"] for c in listing.json()] == [body["id"]]


def test_a_second_user_cannot_see_the_first_users_conversations():
    client, _ = make_client_with_fake_engine()
    token_a = register_and_login(client, email="a@example.com")
    token_b = register_and_login(client, email="b@example.com")

    client.post(
        "/api/v1/conversations", json={"document_id": "doc-1"}, headers=auth_headers(token_a)
    )

    listing_b = client.get("/api/v1/conversations", headers=auth_headers(token_b))
    assert listing_b.json() == []


def test_chat_persists_both_the_question_and_the_answer():
    client, fake_engine = make_client_with_fake_engine(response_text="Grounded answer [1].")
    token = register_and_login(client)
    conversation_id = client.post(
        "/api/v1/conversations", json={"document_id": "doc-1"}, headers=auth_headers(token)
    ).json()["id"]

    response = client.post(
        "/api/v1/chat",
        json={"conversation_id": conversation_id, "question": "What does it say?"},
        headers=auth_headers(token),
    )
    assert response.status_code == 200
    assert response.json()["text"] == "Grounded answer [1]."

    detail = client.get(
        f"/api/v1/conversations/{conversation_id}", headers=auth_headers(token)
    ).json()
    roles = [message["role"] for message in detail["messages"]]
    assert roles == ["user", "assistant"]
    assert detail["messages"][0]["content"] == "What does it say?"
    assert detail["messages"][1]["content"] == "Grounded answer [1]."


def test_chat_stream_persists_both_the_question_and_the_answer():
    client, fake_engine = make_client_with_fake_engine(response_text="Grounded answer [1].")
    token = register_and_login(client)
    conversation_id = client.post(
        "/api/v1/conversations", json={"document_id": "doc-1"}, headers=auth_headers(token)
    ).json()["id"]

    response = client.post(
        "/api/v1/chat/stream",
        json={"conversation_id": conversation_id, "question": "What does it say?"},
        headers=auth_headers(token),
    )
    assert response.status_code == 200
    assert "Grounded answer [1]." in response.text

    detail = client.get(
        f"/api/v1/conversations/{conversation_id}", headers=auth_headers(token)
    ).json()
    roles = [message["role"] for message in detail["messages"]]
    assert roles == ["user", "assistant"]
    assert detail["messages"][1]["content"] == "Grounded answer [1]."


def test_a_second_question_threads_the_first_turn_into_the_llm_as_history():
    client, fake_engine = make_client_with_fake_engine(response_text="Grounded answer [1].")
    token = register_and_login(client)
    conversation_id = client.post(
        "/api/v1/conversations", json={"document_id": "doc-1"}, headers=auth_headers(token)
    ).json()["id"]

    client.post(
        "/api/v1/chat",
        json={"conversation_id": conversation_id, "question": "What does it say?"},
        headers=auth_headers(token),
    )
    client.post(
        "/api/v1/chat",
        json={"conversation_id": conversation_id, "question": "So how does that work?"},
        headers=auth_headers(token),
    )

    # The second call's prompt should carry the first turn as history, ahead
    # of the final user message built for the second question.
    history_roles = [message["role"] for message in fake_engine.llm.last_messages[1:-1]]
    assert history_roles == ["user", "assistant"]
    assert fake_engine.llm.last_messages[1]["content"] == "What does it say?"
    assert fake_engine.llm.last_messages[2]["content"] == "Grounded answer [1]."


def test_chat_on_a_conversation_owned_by_another_user_is_not_found():
    client, _ = make_client_with_fake_engine()
    token_a = register_and_login(client, email="a@example.com")
    token_b = register_and_login(client, email="b@example.com")
    conversation_id = client.post(
        "/api/v1/conversations", json={"document_id": "doc-1"}, headers=auth_headers(token_a)
    ).json()["id"]

    response = client.post(
        "/api/v1/chat",
        json={"conversation_id": conversation_id, "question": "What does it say?"},
        headers=auth_headers(token_b),
    )
    assert response.status_code == 404


def test_rename_and_delete_a_conversation():
    client, _ = make_client_with_fake_engine()
    token = register_and_login(client)
    conversation_id = client.post(
        "/api/v1/conversations", json={"document_id": "doc-1"}, headers=auth_headers(token)
    ).json()["id"]

    renamed = client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "Renamed"},
        headers=auth_headers(token),
    )
    assert renamed.json()["title"] == "Renamed"

    deleted = client.delete(f"/api/v1/conversations/{conversation_id}", headers=auth_headers(token))
    assert deleted.status_code == 204

    listing = client.get("/api/v1/conversations", headers=auth_headers(token))
    assert listing.json() == []


def test_a_global_question_streams_a_summary_not_plain_retrieval():
    # Regression: /chat/stream used to always call stream_answer() (plain
    # top-k retrieval), so "summarize this paper" streamed a weak or
    # abstained answer instead of routing to map-reduce summarisation the
    # way /chat already did - see chat.py's chat_stream() docstring.
    chunks = [make_chunk("real source text", index=0, section_path=["2 Methods"])]
    summaries = [
        SectionSummary(
            marker=1, section="2 Methods", text="summary", chunk_ids=[chunks[0].chunk_id]
        )
    ]
    client, fake_engine = make_client_with_fake_engine(
        store=FakeStore(chunks), summarizer=FakeSummarizer("The paper studies X [1].", summaries)
    )
    token = register_and_login(client)
    conversation_id = client.post(
        "/api/v1/conversations", json={"document_id": "doc-1"}, headers=auth_headers(token)
    ).json()["id"]

    response = client.post(
        "/api/v1/chat/stream",
        json={"conversation_id": conversation_id, "question": "Summarize this paper"},
        headers=auth_headers(token),
    )

    assert response.status_code == 200
    assert "The paper studies X [1]." in response.text
    assert fake_engine.summarizer.calls  # summarize() ran, not just retrieval

    detail = client.get(
        f"/api/v1/conversations/{conversation_id}", headers=auth_headers(token)
    ).json()
    assert detail["messages"][1]["content"] == "The paper studies X [1]."
