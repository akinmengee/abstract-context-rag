"""Shapes for the golden set and the report an evaluation run produces."""

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class QuestionKind(StrEnum):
    SINGLE = "single"  # one section of one paper answers it
    COMPARISON = "comparison"  # two or more papers, each answering its own half
    MULTI_HOP = "multi_hop"  # one hop's answer is needed to find the next one
    ABSTAIN = "abstain"  # the corpus (or the scoped paper) does not contain it


class Evidence(BaseModel):
    """One place an answer has to come from.

    `paper` is an arXiv ID rather than a document ID or origin URL: it is what a
    person writing the golden set actually knows, and a chunk belongs to it when
    its origin contains the ID. Section names repeat across papers ("1
    Introduction"), so a section only counts together with its paper.
    """

    paper: str
    section: str


class GoldenQuestion(BaseModel):
    """One question whose correct behaviour is known in advance.

    `scope` limits retrieval to one paper - "this paper" in a question means
    nothing once a second paper is ingested. None searches the whole corpus.
    """

    question: str
    expected_answer: str
    kind: QuestionKind
    scope: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def is_abstain(self) -> bool:
        return self.kind is QuestionKind.ABSTAIN

    @model_validator(mode="after")
    def _evidence_fits_kind(self) -> "GoldenQuestion":
        # A wrong evidence count silently skews every metric built on it, so a
        # malformed golden file fails at load time instead.
        count = len(self.evidence)
        valid = {
            QuestionKind.SINGLE: count == 1,
            QuestionKind.COMPARISON: count >= 2,
            QuestionKind.MULTI_HOP: count >= 2,
            QuestionKind.ABSTAIN: count == 0,
        }[self.kind]
        if not valid:
            raise ValueError(f"{self.kind.value} question has {count} evidence: {self.question}")
        return self


class RetrievedSection(BaseModel):
    """Where one chunk the LLM saw came from."""

    origin: str
    section: str


class QuestionResult(BaseModel):
    """What the engine actually did with one golden question."""

    question: str
    kind: QuestionKind
    abstained: bool
    abstain_correct: bool
    # Every chunk the engine put in front of the LLM, in rank order.
    retrieved: list[RetrievedSection] = Field(default_factory=list)
    # single questions only: rank of the one expected section.
    hit_rank: int | None = None
    # comparison / multi-hop only: share of the expected evidence that was retrieved.
    evidence_recall: float | None = None
    # The rerank score engine._select_context() compared against score_threshold.
    # None only when nothing was retrieved at all. Lets a threshold miss ("WRONGLY
    # abstained") be told apart from a retrieval miss without rerunning anything.
    top_score: float | None = None
    # None when the judge was disabled, or when the question abstained (nothing to judge).
    faithful: bool | None = None
    # None only when the correctness judge was disabled.
    correct: bool | None = None
    # Queries an agent searched for; empty with agent.mode off.
    searches: list[str] = Field(default_factory=list)
    answer: str = ""


class KindSummary(BaseModel):
    count: int
    abstain_accuracy: float
    evidence_recall: float | None = None
    accuracy: float | None = None


class EvaluationReport(BaseModel):
    """Aggregates plus the config that produced them.

    The config fields are what make two reports comparable: an ablation table is
    several of these lined up, and without them a row is just a number with no
    explanation of what produced it.
    """

    retrieval_mode: str
    reranker_enabled: bool
    agent_mode: str
    k: int

    total_questions: int
    # Over single questions, so these stay comparable with the phase 2 table.
    recall_at_k: float
    mrr: float
    abstain_accuracy: float
    # Over comparison and multi-hop questions.
    evidence_recall: float
    # Chunks per answer: an engine that retrieves more can raise evidence recall
    # just by retrieving more, and this keeps that visible.
    mean_context_chunks: float
    faithfulness: float | None = None
    accuracy: float | None = None
    by_kind: dict[str, KindSummary] = Field(default_factory=dict)

    results: list[QuestionResult] = Field(default_factory=list)
