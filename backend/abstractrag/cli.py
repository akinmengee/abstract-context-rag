"""abstractrag command line interface.

Calls the same engine the API calls, so anything reachable over HTTP is also
reachable from a terminal or a script - useful for ingesting a paper set or
running an evaluation without starting a server.
"""

import json
from pathlib import Path

import typer

from abstractrag.core.container import get_engine
from abstractrag.core.logging import setup_logging
from abstractrag.rag.ingestion.base import SourceInput

app = typer.Typer(help="Local RAG engine for research papers and Wikipedia.", no_args_is_help=True)


@app.command()
def ingest(
    arxiv: str = typer.Option(None, "--arxiv", help="arXiv ID, e.g. 2005.11401"),
    wiki: str = typer.Option(None, "--wiki", help="Wikipedia article title or URL"),
    file: Path = typer.Option(None, "--file", exists=True, help="Path to a local PDF"),
) -> None:
    """Ingest one source: an arXiv paper, a Wikipedia article, or a local PDF."""
    setup_logging()
    source = SourceInput(
        arxiv_id=arxiv,
        wikipedia=wiki,
        file_name=file.name if file else None,
        file_bytes=file.read_bytes() if file else None,
    )
    result = get_engine().ingest(source)
    typer.echo(f"{result.title}\n  {result.chunk_count} chunks  |  id {result.document_id}")


@app.command()
def ask(
    question: str,
    document_id: str = typer.Option(None, "--document-id", help="Limit to one document"),
) -> None:
    """Ask a question and print the grounded answer with its citations."""
    setup_logging()
    answer = get_engine().answer(question, document_id)
    typer.echo(answer.text)
    for citation in answer.citations:
        location = citation.section or (f"page {citation.page}" if citation.page else "")
        typer.echo(f"  [{citation.marker}] {citation.title} {location} — {citation.origin}")


@app.command()
def documents() -> None:
    """List everything that has been ingested."""
    setup_logging()
    engine = get_engine()
    engine.store.ensure_collection()
    for document in engine.store.list_documents():
        typer.echo(f"{document['title']}  ({document['chunk_count']} chunks)")
        typer.echo(f"  {document['document_id']}  {document['origin']}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run("abstractrag.main:app", host=host, port=port, reload=reload)


@app.command("export-openapi")
def export_openapi(
    output: Path = typer.Option(Path("shared/openapi.json"), "--output"),
) -> None:
    """Write the OpenAPI schema that the React and Flutter clients are generated from."""
    from abstractrag.main import create_app

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(create_app().openapi(), indent=2), encoding="utf-8")
    typer.echo(f"wrote {output}")


if __name__ == "__main__":
    app()
