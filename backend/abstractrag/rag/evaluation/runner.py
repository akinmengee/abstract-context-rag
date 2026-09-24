"""Runs a golden set through the engine and aggregates the result into a report.

Deliberately takes the engine and the judge as arguments rather than building
them: the unit tests drive this with canned answers and no LLM at all, and the
CLI is the only place that wires the real ones in.
"""

import json
from collections.abc import Callable
from pathlib import Path

from abstractrag.core.logging import get_logger
from abstractrag.rag.evaluation.metrics import first_hit_rank, recall_at_k, reciprocal_rank
from abstractrag.rag.evaluation.models import EvaluationReport, GoldenQuestion, QuestionResult

logger = get_logger(__name__)

GOLDEN_DIR = Path(__file__).parent / "golden"
DEFAULT_GOLDEN_SET = GOLDEN_DIR / "rag_paper.json"

# (answer_text, context_text) -> is every claim supported by the context?
Judge = Callable[[str, str], bool]


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    with path.open(encoding="utf-8") as handle:
        return [GoldenQuestion.model_validate(entry) for entry in json.load(handle)]


def run_evaluation(
    engine,
    questions: list[GoldenQuestion],
    judge: Judge | None = None,
    retrieval_only: bool = False,
) -> EvaluationReport:
    """Score a golden set.

    `retrieval_only=True` skips generation entirely (engine.preview_retrieval()
    instead of engine.answer()) and forces the judge off with it - generation is
    the slow part (tens of seconds per question) and ablation runs that compare
    retrieval configs do not need an answer, only the ranked chunks.
    """
    if retrieval_only:
        judge = None

    settings = engine.settings
    k = settings.retrieval.context_size
    results: list[QuestionResult] = []

    for index, question in enumerate(questions, start=1):
        logger.info("[%d/%d] %s", index, len(questions), question.question)

        if retrieval_only:
            chunks, sufficient = engine.preview_retrieval(question.question)
            abstained, answer_text = not sufficient, ""
        else:
            answer = engine.answer(question.question)
            chunks, abstained, answer_text = answer.used_chunks, answer.abstained, answer.text

        sections = [chunk.chunk.metadata.section or "" for chunk in chunks]

        hit_rank = (
            first_hit_rank(sections, question.expected_section)
            if question.expected_section
            else None
        )

        faithful = None
        if judge and not abstained:
            context = "\n\n".join(chunk.chunk.text for chunk in chunks)
            faithful = judge(answer_text, context)

        results.append(
            QuestionResult(
                question=question.question,
                abstained=abstained,
                abstain_correct=abstained == question.is_abstain,
                retrieved_sections=sections,
                hit_rank=hit_rank,
                top_score=chunks[0].effective_score if chunks else None,
                faithful=faithful,
                answer=answer_text,
            )
        )

    return EvaluationReport(
        retrieval_mode=settings.retrieval.mode,
        reranker_enabled=settings.reranker.enabled,
        k=k,
        total_questions=len(questions),
        recall_at_k=_mean(
            [
                recall_at_k(result.retrieved_sections, question.expected_section, k)
                for question, result in zip(questions, results, strict=True)
                if question.expected_section
            ]
        ),
        mrr=_mean(
            [
                reciprocal_rank(result.retrieved_sections, question.expected_section)
                for question, result in zip(questions, results, strict=True)
                if question.expected_section
            ]
        ),
        abstain_accuracy=_mean([result.abstain_correct for result in results]),
        faithfulness=_mean([r.faithful for r in results if r.faithful is not None])
        if judge
        else None,
        results=results,
    )


def _mean(values: list[bool] | list[float]) -> float:
    """Mean of a possibly empty list - an empty slice scores 0.0, not a crash."""
    return sum(values) / len(values) if values else 0.0
