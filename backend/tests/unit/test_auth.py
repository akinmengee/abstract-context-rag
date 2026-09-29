"""HTTP-layer tests for registration and login - the first FastAPI
TestClient tests in this repo, so the dependency-override pattern used here
(override the callable passed to Depends(), never the Annotated[...] alias;
an in-memory sqlite:// engine needs StaticPool or each Session() gets its
own separate empty database) is worth getting right once and reusing."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from abstractrag.accounts import models  # noqa: F401 - registers tables on SQLModel.metadata
from abstractrag.api.dependencies import get_session
from abstractrag.main import app


def make_client() -> TestClient:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_register_returns_a_bearer_token():
    response = make_client().post(
        "/api/v1/auth/register", json={"email": "a@example.com", "password": "hunter2222"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["email"] == "a@example.com"
    assert body["access_token"]


def test_registering_the_same_email_twice_is_rejected():
    client = make_client()
    client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "hunter2222"})
    response = client.post(
        "/api/v1/auth/register", json={"email": "a@example.com", "password": "different1"}
    )
    assert response.status_code == 409


def test_login_returns_a_bearer_token_for_the_right_password():
    client = make_client()
    client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "hunter2222"})
    response = client.post(
        "/api/v1/auth/login", json={"email": "a@example.com", "password": "hunter2222"}
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_login_with_the_wrong_password_is_rejected():
    client = make_client()
    client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "hunter2222"})
    response = client.post(
        "/api/v1/auth/login", json={"email": "a@example.com", "password": "wrong"}
    )
    assert response.status_code == 401


def test_login_with_an_unknown_email_is_rejected():
    response = make_client().post(
        "/api/v1/auth/login", json={"email": "nobody@example.com", "password": "hunter2222"}
    )
    assert response.status_code == 401


def test_an_authenticated_endpoint_rejects_a_missing_token():
    response = make_client().get("/api/v1/conversations")
    assert response.status_code == 401


def test_an_authenticated_endpoint_rejects_a_garbage_token():
    client = make_client()
    response = client.get(
        "/api/v1/conversations", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401
