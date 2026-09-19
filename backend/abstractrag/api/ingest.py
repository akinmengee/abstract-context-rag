"""Ingestion endpoints.

Both routes end in the same engine call: one takes an identifier to download,
the other takes the file itself.
"""

from fastapi import APIRouter, File, UploadFile

from abstractrag.api.dependencies import EngineDep
from abstractrag.api.schemas import IngestRequest
from abstractrag.rag.ingestion.base import SourceInput
from abstractrag.rag.models import IngestResult

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResult)
def ingest(request: IngestRequest, engine: EngineDep) -> IngestResult:
    """Ingest by arXiv ID or Wikipedia title/URL."""
    return engine.ingest(request.to_source_input())


@router.post("/ingest/upload", response_model=IngestResult)
def ingest_upload(engine: EngineDep, file: UploadFile = File(...)) -> IngestResult:
    """Ingest an uploaded PDF."""
    return engine.ingest(SourceInput(file_name=file.filename, file_bytes=file.file.read()))
