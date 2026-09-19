"""Collects every route under a single router mounted at /api/v1."""

from fastapi import APIRouter

from abstractrag.api import chat, documents, health, ingest

router = APIRouter(prefix="/api/v1")
router.include_router(health.router)
router.include_router(ingest.router)
router.include_router(documents.router)
router.include_router(chat.router)
