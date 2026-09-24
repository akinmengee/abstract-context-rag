"""Shapes for the golden set and the report an evaluation run produces."""

from pydantic import BaseModel, Field


class GoldenQuestion(BaseModel):
    """One question whose correct behaviour is known in advance.

    `expected_section` is the paper section the answer lives in - the ground
    truth for retrieval metrics. It is None for abstain questions, where the
    correct behaviour is to retrieve nothing good enough to answer from.
    """

    question: str
    expected_answer: str
    expected_section: str | None = None
    is_abstain: bool = False


class QuestionResult(BaseModel):
    """What the engine actually did with one golden question."""

    question: str
    abstained: bool
    abstain_correct: bool
    # Section names of the chunks the engine put in front of the LLM, in rank order.
    retrieved_sections: list[str] = Field(default_factory=list)
    hit_rank: int | None = None
    # The rerank score engine._select_context() compared against score_threshold.
    # None only when nothing was retrieved at all. Lets a threshold miss ("WRONGLY
    # abstained") be told apart from a retrieval miss without rerunning anything.
    top_score: float | None = None
    # None when the judge was disabled, or when the question abstained (nothing to judge).
    faithful: bool | None = None
    answer: str = ""


class EvaluationReport(BaseModel):
    """Aggregates plus the config that produced them.

    The config fields are what make two reports comparable: an ablation table is
    several of these lined up, and without them a row is just a number with no
    explanation of what produced it.
    """

    retrieval_mode: str
    reranker_enabled: bool
    k: int

    total_questions: int
    recall_at_k: float
    mrr: float
    abstain_accuracy: float
    faithfulness: float | None = None

    results: list[QuestionResult] = Field(default_factory=list)
