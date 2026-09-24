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
from abstractrag.rag.evaluation.judge import LlmJudge
from abstractrag.rag.evaluation.models import EvaluationReport
from abstractrag.rag.evaluation.runner import DEFAULT_GOLDEN_SET, load_golden_set, run_evaluation
from abstractrag.rag.ingestion.base import SourceInput
from abstractrag.rag.models import ClaimVerdict, VerifiedClaim

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

    _print_verification(answer.verified_claims)


@app.command()
def summarize(
    document_id: str = typer.Option(..., "--document-id", help="Document to summarise"),
    question: str = typer.Option(
        None, "--question", help="Focus the summary, e.g. 'What is the main contribution?'"
    ),
) -> None:
    """Summarise a whole document: one LLM call per section, plus one. Minutes, not seconds."""
    setup_logging()
    answer = get_engine().summarize(question, document_id)
    typer.echo(answer.text)
    for citation in answer.citations:
        typer.echo(f"  [{citation.marker}] {citation.section or citation.title}")

    _print_verification(answer.verified_claims)


def _print_verification(claims: list[VerifiedClaim]) -> None:
    """Report the count, then only the claims that failed - clean ones are noise."""
    if not claims:
        return

    supported = sum(claim.verdict is ClaimVerdict.SUPPORTED for claim in claims)
    typer.echo(f"\nverification: {supported}/{len(claims)} claims supported")

    for claim in claims:
        if claim.verdict is ClaimVerdict.SUPPORTED:
            continue
        markers = " ".join(f"[{marker}]" for marker in claim.markers)
        reason = f" — {claim.reason}" if claim.reason else ""
        typer.echo(f'  ! {claim.verdict.value} {markers}: "{claim.text}"{reason}')


@app.command()
def documents() -> None:
    """List everything that has been ingested."""
    setup_logging()
    engine = get_engine()
    engine.store.ensure_collection()
    for document in engine.store.list_documents():
        typer.echo(f"{document['title']}  ({document['chunk_count']} chunks)")
        typer.echo(f"  {document['document_id']}  {document['origin']}")


@app.command("eval")
def evaluate(
    golden: Path = typer.Option(
        DEFAULT_GOLDEN_SET, "--golden", exists=True, help="Golden set JSON"
    ),
    no_judge: bool = typer.Option(
        False, "--no-judge", help="Skip the faithfulness judge (one LLM call per answer)"
    ),
    retrieval_only: bool = typer.Option(
        False,
        "--retrieval-only",
        help="Skip generation entirely (recall@k/MRR/abstain only, no LLM call at all)",
    ),
    json_out: Path = typer.Option(None, "--json", help="Write the raw report for comparisons"),
) -> None:
    """Score the golden set: retrieval quality, abstain correctness, faithfulness.

    Retrieval modes are compared by running this once per mode, e.g.
    ACR_RETRIEVAL__MODE=dense abstractrag eval --retrieval-only --json dense.json
    """
    setup_logging()
    engine = get_engine()
    questions = load_golden_set(golden)
    judge = None if (no_judge or retrieval_only) else LlmJudge(engine.llm)

    report = run_evaluation(engine, questions, judge=judge, retrieval_only=retrieval_only)
    _print_report(report)

    if json_out:
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"\nwrote {json_out}")


def _print_report(report: EvaluationReport) -> None:
    typer.echo(
        f"\nmode={report.retrieval_mode} rerank={report.reranker_enabled} k={report.k}\n"
    )
    for result in report.results:
        if result.abstain_correct:
            verdict = "abstained" if result.abstained else "answered"
        else:
            verdict = "WRONGLY abstained" if result.abstained else "SHOULD have abstained"
        rank = f"rank {result.hit_rank}" if result.hit_rank else "section miss"
        score = "score=none" if result.top_score is None else f"score={result.top_score:.2f}"
        faithful = "" if result.faithful is None else f"  faithful={result.faithful}"
        typer.echo(f"  {verdict:<22} {rank:<14} {score:<12}{faithful}  {result.question[:60]}")

    typer.echo(
        f"\nrecall@{report.k}: {report.recall_at_k:.2f}"
        f"   MRR: {report.mrr:.2f}"
        f"   abstain accuracy: {report.abstain_accuracy:.2f}"
        + (
            ""
            if report.faithfulness is None
            else f"   faithfulness: {report.faithfulness:.2f}"
        )
    )


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
