"""Collects every route under a single router mounted at /api/v1."""

from fastapi import APIRouter

from abstractrag.api import auth, chat, conversations, documents, health, ingest

router = APIRouter(prefix="/api/v1")
router.include_router(health.router)
router.include_router(auth.router)
router.include_router(ingest.router)
router.include_router(documents.router)
router.include_router(conversations.router)
router.include_router(chat.router)
