"""abstractrag command line interface.

Calls the same engine the API calls, so anything reachable over HTTP is also
reachable from a terminal or a script - useful for ingesting a paper set or
running an evaluation without starting a server.
"""

import json
import time
from enum import StrEnum
from pathlib import Path

import typer

from abstractrag.core.container import get_engine
from abstractrag.core.logging import setup_logging
from abstractrag.rag.evaluation.judge import CorrectnessJudge, LlmJudge
from abstractrag.rag.evaluation.models import EvaluationReport
from abstractrag.rag.evaluation.runner import (
    MissingPapersError,
    default_golden_sets,
    load_golden_set,
    run_evaluation,
    select_split,
)
from abstractrag.rag.ingestion.base import SourceInput
from abstractrag.rag.models import ClaimVerdict, VerifiedClaim
from abstractrag.rag.query.router import is_global_question


class SplitChoice(StrEnum):
    EVAL = "eval"
    TRAIN = "train"
    ALL = "all"


class EntryChoice(StrEnum):
    ANSWER = "answer"
    ASK = "ask"


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
    if result.tree_nodes:
        typer.echo(f"  {result.tree_nodes} tree nodes")


@app.command("build-tree")
def build_tree(
    document_id: str = typer.Option(None, "--document-id", help="One document; default: all"),
) -> None:
    """Build RAPTOR summary trees: one LLM call per cluster, once per document."""
    setup_logging()
    engine = get_engine()
    documents = engine.store.list_documents()
    ids = [document_id] if document_id else [document["document_id"] for document in documents]
    for doc_id in ids:
        started = time.perf_counter()
        count = engine.build_tree(doc_id)
        typer.echo(f"{doc_id}: {count} tree nodes in {time.perf_counter() - started:.0f}s")


@app.command()
def ask(
    question: str,
    document_id: str = typer.Option(None, "--document-id", help="Limit to one document"),
) -> None:
    """Ask a question and print the grounded answer with its citations.

    A global question (e.g. "summarize this paper") with a --document-id is
    routed to a full-document summary instead of a retrieval answer - see
    RagEngine.ask(). That can take several minutes rather than seconds.
    """
    setup_logging()
    if document_id and is_global_question(question):
        typer.echo(
            "Global question detected - summarising the whole document "
            "(about half a minute from a RAPTOR tree, minutes with map-reduce).",
            err=True,
        )
    answer = get_engine().ask(question, document_id)
    typer.echo(answer.text)
    for citation in answer.citations:
        location = citation.section or (f"page {citation.page}" if citation.page else "")
        typer.echo(f"  [{citation.marker}] {citation.title} {location} — {citation.origin}")

    if answer.agent_steps:
        typer.echo("\nsearches:")
        for step in answer.agent_steps:
            typer.echo(f"  kept {step.kept}/{step.retrieved}  {step.query}")

    _print_verification(answer.verified_claims)


@app.command()
def summarize(
    document_id: str = typer.Option(..., "--document-id", help="Document to summarise"),
    question: str = typer.Option(
        None, "--question", help="Focus the summary, e.g. 'What is the main contribution?'"
    ),
) -> None:
    """Summarise a whole document from its RAPTOR tree, or section by section (map-reduce).

    summarization.method raptor needs a built tree (build-tree, or
    raptor.build_on_ingest) and takes about half a minute; map-reduce takes
    one LLM call per section, i.e. minutes.
    """
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
    golden: list[Path] = typer.Option(
        None,
        "--golden",
        exists=True,
        help="Golden set JSON; repeat for several. Default: every file in the golden dir",
    ),
    no_judge: bool = typer.Option(
        False,
        "--no-judge",
        help="Skip the faithfulness and correctness judges (two LLM calls per answer)",
    ),
    retrieval_only: bool = typer.Option(
        False,
        "--retrieval-only",
        help="Skip generation entirely (retrieval and abstain metrics only, no LLM call)",
    ),
    json_out: Path = typer.Option(None, "--json", help="Write the raw report for comparisons"),
    split: SplitChoice = typer.Option(
        SplitChoice.EVAL,
        "--split",
        help="Which golden questions to score. Reported numbers come from eval only",
    ),
    entry: EntryChoice = typer.Option(
        EntryChoice.ANSWER,
        "--entry",
        help="answer: retrieval for every question. "
        "ask: the router, so global questions are summarised",
    ),
) -> None:
    """Score golden sets: retrieval quality, evidence recall, abstains, faithfulness, accuracy.

    Retrieval modes are compared by running this once per mode, e.g.
    ACR_RETRIEVAL__MODE=dense abstractrag eval --retrieval-only --json dense.json
    The phase 2 ablation table was measured on rag_paper.json alone.
    """
    setup_logging()
    engine = get_engine()
    questions = select_split(
        [q for path in golden or default_golden_sets() for q in load_golden_set(path)],
        split.value,
    )
    if not questions:
        typer.echo(f"error: no golden questions in split {split.value!r}", err=True)
        raise typer.Exit(1)
    judged = not (no_judge or retrieval_only)

    try:
        report = run_evaluation(
            engine,
            questions,
            judge=LlmJudge(engine.llm) if judged else None,
            correctness=CorrectnessJudge(engine.llm) if judged else None,
            retrieval_only=retrieval_only,
            entry=entry.value,
        )
    except MissingPapersError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    _print_report(report)

    if json_out:
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"\nwrote {json_out}")


def _print_report(report: EvaluationReport) -> None:
    typer.echo(
        f"\nmode={report.retrieval_mode} rerank={report.reranker_enabled} "
        f"agent={report.agent_mode} entry={report.entry} summary={report.summary_method} "
        f"tree={report.tree_retrieval} k={report.k}\n"
    )
    for result in report.results:
        if result.abstain_correct:
            verdict = "abstained" if result.abstained else "answered"
        else:
            verdict = "WRONGLY abstained" if result.abstained else "SHOULD have abstained"
        if result.evidence_recall is not None:
            found = f"evidence {result.evidence_recall:.2f}"
        elif result.hit_rank:
            found = f"rank {result.hit_rank}"
        else:
            found = "-" if result.kind.value in ("abstain", "global") else "section miss"
        score = "score=none" if result.top_score is None else f"score={result.top_score:.2f}"
        faithful = "" if result.faithful is None else f"  faithful={result.faithful}"
        correct = "" if result.correct is None else f"  correct={result.correct}"
        searches = f"  searches={len(result.searches)}" if result.searches else ""
        typer.echo(
            f"  {result.kind.value:<10} {verdict:<22} {found:<14} {score:<12}"
            f"{result.seconds:>5.0f}s{faithful}{correct}{searches}  {result.question[:60]}"
        )

    typer.echo(
        f"\nrecall@{report.k}: {report.recall_at_k:.2f}"
        f"   MRR: {report.mrr:.2f}"
        f"   abstain accuracy: {report.abstain_accuracy:.2f}"
        f"   chunks/answer: {report.mean_context_chunks:.1f}"
        + (
            ""
            if report.evidence_recall is None
            else f"   evidence recall: {report.evidence_recall:.2f}"
        )
        + ("" if report.faithfulness is None else f"   faithfulness: {report.faithfulness:.2f}")
        + ("" if report.accuracy is None else f"   accuracy: {report.accuracy:.2f}")
        + f"   secs/question: {report.mean_seconds:.1f}"
        + (
            ""
            if report.supported_claims is None
            else f"   supported claims: {report.supported_claims:.2f}"
        )
    )

    typer.echo(
        f"\n  {'kind':<10} {'n':>3}  {'evidence':>8}  {'abstain':>7}  {'accuracy':>8}"
        f"  {'secs':>6}  {'supported':>9}"
    )
    for kind, summary in report.by_kind.items():
        evidence = "-" if summary.evidence_recall is None else f"{summary.evidence_recall:.2f}"
        accuracy = "-" if summary.accuracy is None else f"{summary.accuracy:.2f}"
        supported = "-" if summary.supported_claims is None else f"{summary.supported_claims:.2f}"
        typer.echo(
            f"  {kind:<10} {summary.count:>3}  {evidence:>8}  "
            f"{summary.abstain_accuracy:>7.2f}  {accuracy:>8}  "
            f"{summary.mean_seconds:>6.1f}  {supported:>9}"
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
