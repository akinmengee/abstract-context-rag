"""Runs a golden set through the engine and aggregates the result into a report.

Deliberately takes the engine and the judges as arguments rather than building
them: the unit tests drive this with canned answers and no LLM at all, and the
CLI is the only place that wires the real ones in.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

from abstractrag.core.logging import get_logger
from abstractrag.rag.evaluation.metrics import (
    evidence_recall,
    first_hit_rank,
    recall_at_k,
    reciprocal_rank,
)
from abstractrag.rag.evaluation.models import (
    EvaluationReport,
    GoldenQuestion,
    KindSummary,
    QuestionKind,
    QuestionResult,
    RetrievedSection,
    Split,
)
from abstractrag.rag.models import ClaimVerdict, VerifiedClaim

logger = get_logger(__name__)

GOLDEN_DIR = Path(__file__).parent / "golden"

# (answer_text, context_text) -> is every claim supported by the context?
Judge = Callable[[str, str], bool]
# (question, expected_answer, answer_text) -> does the answer get it right?
CorrectnessJudge = Callable[[str, str, str], bool]

_MULTI_EVIDENCE = {QuestionKind.COMPARISON, QuestionKind.MULTI_HOP}


class MissingPapersError(Exception):
    """A golden set names a paper that is not in the store.

    Raised before any question runs: otherwise every question touching that
    paper scores zero, which reads exactly like a retrieval failure.
    """

    def __init__(self, papers: list[str]) -> None:
        self.papers = papers
        commands = "; ".join(f"abstractrag ingest --arxiv {paper}" for paper in papers)
        super().__init__(f"not ingested: {', '.join(papers)} - run: {commands}")


def default_golden_sets() -> list[Path]:
    return sorted(GOLDEN_DIR.glob("*.json"))


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    with path.open(encoding="utf-8") as handle:
        return [GoldenQuestion.model_validate(entry) for entry in json.load(handle)]


def select_split(questions: list[GoldenQuestion], split: str) -> list[GoldenQuestion]:
    """The questions of one split, or all of them for "all"."""
    if split == "all":
        return list(questions)
    return [question for question in questions if question.split == Split(split)]


def resolve_papers(questions: list[GoldenQuestion], documents: list[dict]) -> dict[str, str]:
    """arXiv ID -> document ID, for every paper a scope or an evidence names."""
    papers = {question.scope for question in questions if question.scope} | {
        item.paper for question in questions for item in question.evidence
    }
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for paper in sorted(papers):
        document = next((doc for doc in documents if paper in doc["origin"]), None)
        if document is None:
            missing.append(paper)
        else:
            resolved[paper] = document["document_id"]
    if missing:
        raise MissingPapersError(missing)
    return resolved


def run_evaluation(
    engine,
    questions: list[GoldenQuestion],
    judge: Judge | None = None,
    correctness: CorrectnessJudge | None = None,
    retrieval_only: bool = False,
    entry: str = "answer",
) -> EvaluationReport:
    """Score a golden set.

    `retrieval_only=True` skips generation entirely (engine.preview_retrieval()
    instead of engine.answer()) and forces both judges off with it - generation
    is the slow part (tens of seconds per question) and ablation runs that
    compare retrieval configs do not need an answer, only the ranked chunks.

    `entry="ask"` goes through the router, so global questions reach
    summarize() the way a user's would; "answer" is retrieval for everything.
    """
    if retrieval_only:
        judge = correctness = None

    papers = resolve_papers(questions, engine.store.list_documents())
    results: list[QuestionResult] = []

    for index, question in enumerate(questions, start=1):
        logger.info("[%d/%d] %s", index, len(questions), question.question)
        document_id = papers[question.scope] if question.scope else None

        started = time.perf_counter()
        supported = None
        if retrieval_only:
            chunks, sufficient = engine.preview_retrieval(question.question, document_id)
            abstained, answer_text, searches = not sufficient, "", []
        else:
            call = engine.ask if entry == "ask" else engine.answer
            answer = call(question.question, document_id)
            chunks, abstained, answer_text = answer.used_chunks, answer.abstained, answer.text
            searches = [step.query for step in answer.agent_steps]
            supported = _supported_share(answer.verified_claims)
        seconds = time.perf_counter() - started

        retrieved = [
            RetrievedSection(
                origin=chunk.chunk.metadata.origin, section=chunk.chunk.metadata.section or ""
            )
            for chunk in chunks
        ]

        # A summary has no retrieved chunks to be faithful to; verification
        # (supported_claims) is what checks it against its sources.
        faithful = None
        if judge and not abstained and chunks:
            faithful = judge(answer_text, "\n\n".join(chunk.chunk.text for chunk in chunks))

        results.append(
            QuestionResult(
                question=question.question,
                kind=question.kind,
                abstained=abstained,
                abstain_correct=abstained == question.is_abstain,
                retrieved=retrieved,
                hit_rank=first_hit_rank(retrieved, question.evidence[0])
                if question.kind is QuestionKind.SINGLE
                else None,
                evidence_recall=evidence_recall(retrieved, question.evidence)
                if question.kind in _MULTI_EVIDENCE
                else None,
                top_score=chunks[0].effective_score if chunks else None,
                faithful=faithful,
                correct=_judge_correctness(question, abstained, answer_text, correctness),
                seconds=seconds,
                supported_claims=supported,
                searches=searches,
                answer=answer_text,
            )
        )

    return _report(engine.settings, questions, results, judge, correctness, entry)


def _judge_correctness(
    question: GoldenQuestion,
    abstained: bool,
    answer_text: str,
    correctness: CorrectnessJudge | None,
) -> bool | None:
    """Abstaining is decided without the LLM: it is right exactly when the
    question is an abstain question, and never a correct answer otherwise."""
    if correctness is None:
        return None
    if question.is_abstain or abstained:
        return question.is_abstain and abstained
    return correctness(question.question, question.expected_answer, answer_text)


def _report(
    settings,
    questions: list[GoldenQuestion],
    results: list[QuestionResult],
    judge: Judge | None,
    correctness: CorrectnessJudge | None,
    entry: str,
) -> EvaluationReport:
    k = settings.retrieval.context_size
    pairs = list(zip(questions, results, strict=True))
    singles = [(q, r) for q, r in pairs if q.kind is QuestionKind.SINGLE]

    return EvaluationReport(
        retrieval_mode=settings.retrieval.mode,
        reranker_enabled=settings.reranker.enabled,
        agent_mode=settings.agent.mode,
        entry=entry,
        summary_method=settings.summarization.method,
        tree_retrieval=settings.retrieval.include_tree_nodes,
        k=k,
        total_questions=len(questions),
        recall_at_k=_mean([recall_at_k(r.retrieved, q.evidence[0], k) for q, r in singles]),
        mrr=_mean([reciprocal_rank(r.retrieved, q.evidence[0]) for q, r in singles]),
        abstain_accuracy=_mean([r.abstain_correct for r in results]),
        evidence_recall=_mean_or_none([r.evidence_recall for r in results]),
        mean_context_chunks=_mean([len(r.retrieved) for r in results if not r.abstained]),
        mean_seconds=_mean([r.seconds for r in results]),
        supported_claims=_mean_or_none([r.supported_claims for r in results]),
        faithfulness=_mean_or_none([r.faithful for r in results]) if judge else None,
        accuracy=_mean([r.correct for r in results if r.correct is not None])
        if correctness
        else None,
        by_kind=_by_kind(results, correctness is not None),
        results=results,
    )


def _by_kind(results: list[QuestionResult], judged: bool) -> dict[str, KindSummary]:
    """Per-kind numbers: a multi-hop score averaged in with single-paper
    questions would disappear into the mean."""
    summaries: dict[str, KindSummary] = {}
    for kind in QuestionKind:
        group = [r for r in results if r.kind is kind]
        if not group:
            continue
        summaries[kind.value] = KindSummary(
            count=len(group),
            abstain_accuracy=_mean([r.abstain_correct for r in group]),
            evidence_recall=_mean([r.evidence_recall for r in group])
            if kind in _MULTI_EVIDENCE
            else None,
            accuracy=_mean([r.correct for r in group]) if judged else None,
            mean_seconds=_mean([r.seconds for r in group]),
            supported_claims=_mean_or_none([r.supported_claims for r in group]),
        )
    return summaries


def _mean(values: list[bool] | list[float] | list[int]) -> float:
    """Mean of a possibly empty list - an empty slice scores 0.0, not a crash."""
    return sum(values) / len(values) if values else 0.0


def _supported_share(claims: list[VerifiedClaim]) -> float | None:
    if not claims:
        return None
    return sum(claim.verdict is ClaimVerdict.SUPPORTED for claim in claims) / len(claims)


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return _mean(present) if present else None
