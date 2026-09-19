"""Read-only view of what has been ingested."""

from fastapi import APIRouter

from abstractrag.api.dependencies import EngineDep
from abstractrag.api.schemas import DocumentSummary

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents(engine: EngineDep) -> list[DocumentSummary]:
    engine.store.ensure_collection()
    return [DocumentSummary(**document) for document in engine.store.list_documents()]
