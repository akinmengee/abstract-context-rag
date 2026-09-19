"""FastAPI application.

The API is a thin shell: it validates input, calls the engine, and maps domain
errors to status codes. No RAG logic lives here.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from abstractrag import __version__
from abstractrag.api.router import router as api_router
from abstractrag.core.errors import (
    FetchError,
    InvalidInputError,
    LLMError,
    ParseError,
    RagError,
    SourceNotFoundError,
)
from abstractrag.core.logging import setup_logging

_STATUS_BY_ERROR: list[tuple[type[RagError], int]] = [
    (InvalidInputError, 400),
    (SourceNotFoundError, 404),
    (ParseError, 422),
    (FetchError, 502),
    (LLMError, 503),
]


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(
        title="abstract-context-rag",
        version=__version__,
        description="Local RAG engine for grounded Q&A over research papers and Wikipedia.",
    )

    # The React dev server and the phone on the same LAN call this API directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.exception_handler(RagError)
    def handle_rag_error(_request: Request, exc: RagError) -> JSONResponse:
        status = next((code for error, code in _STATUS_BY_ERROR if isinstance(exc, error)), 500)
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    return app


app = create_app()
