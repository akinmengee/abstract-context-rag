"""Health endpoints.

/health answers whether the process is up (used by Docker).
/health/dependencies also checks Qdrant and the llama.cpp server.
"""

from fastapi import APIRouter

from abstractrag.api.dependencies import EngineDep

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/dependencies")
def dependencies(engine: EngineDep) -> dict[str, bool]:
    return engine.health()
