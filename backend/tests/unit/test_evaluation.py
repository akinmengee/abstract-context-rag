"""Evaluation metrics and the runner that turns a golden set into a report."""

from abstractrag.core.config import RetrievalSettings, Settings
from abstractrag.rag.evaluation.judge import LlmJudge
from abstractrag.rag.evaluation.metrics import first_hit_rank, recall_at_k, reciprocal_rank
from abstractrag.rag.evaluation.models import GoldenQuestion
from abstractrag.rag.evaluation.runner import run_evaluation
from abstractrag.rag.models import Answer, RetrievedChunk
from tests.conftest import make_chunk

SECTIONS = ["1 Introduction", "2.2 Retriever: DPR", "3 Results"]


class TestFirstHitRank:
    def test_returns_one_based_rank_of_the_expected_section(self):
        assert first_hit_rank(SECTIONS, "2.2 Retriever: DPR") == 2

    def test_returns_none_when_the_section_was_never_retrieved(self):
        assert first_hit_rank(SECTIONS, "5 Conclusion") is None

    def test_ignores_case_and_surrounding_whitespace(self):
        # Docling section titles vary in spacing; the golden set is hand-written.
        assert first_hit_rank(SECTIONS, "  2.2 retriever: dpr ") == 2

    def test_returns_none_for_an_empty_ranking(self):
        assert first_hit_rank([], "2.2 Retriever: DPR") is None


class TestRecallAtK:
    def test_hits_when_the_section_is_within_k(self):
        assert recall_at_k(SECTIONS, "2.2 Retriever: DPR", k=2)

    def test_misses_when_the_section_ranks_below_k(self):
        assert not recall_at_k(SECTIONS, "3 Results", k=2)

    def test_misses_when_the_section_was_never_retrieved(self):
        assert not recall_at_k(SECTIONS, "5 Conclusion", k=3)


class TestReciprocalRank:
    def test_is_one_over_the_first_hit_position(self):
        assert reciprocal_rank(SECTIONS, "2.2 Retriever: DPR") == 0.5

    def test_is_one_when_the_expected_section_ranks_first(self):
        assert reciprocal_rank(SECTIONS, "1 Introduction") == 1.0

    def test_is_zero_when_the_section_was_never_retrieved(self):
        assert reciprocal_rank(SECTIONS, "5 Conclusion") == 0.0


def answered(text: str, sections: list[str]) -> Answer:
    """An Answer whose retrieved context came from the given sections, in rank order."""
    return Answer(
        text=text,
        abstained=False,
        used_chunks=[
            RetrievedChunk(chunk=make_chunk(f"chunk {i}", index=i, section=section), score=1.0)
            for i, section in enumerate(sections)
        ],
    )


def abstained(sections: list[str] | None = None) -> Answer:
    return Answer(
        text="This source does not contain that information.",
        abstained=True,
        used_chunks=[
            RetrievedChunk(chunk=make_chunk(f"chunk {i}", index=i, section=section), score=0.1)
            for i, section in enumerate(sections or [])
        ],
    )


class FakeEngine:
    """Replays a canned Answer per question, in the order the runner asks.

    Both answer() and preview_retrieval() draw from the same canned list, so a
    test can build one set of expectations with answered()/abstained() and use
    it to drive either path - only call counts tell them apart.
    """

    def __init__(self, answers: list[Answer]) -> None:
        self.answers = list(answers)
        self.settings = Settings(retrieval=RetrievalSettings(context_size=3))
        self.answer_calls = 0
        self.preview_calls = 0

    def answer(self, question: str, document_id: str | None = None) -> Answer:
        self.answer_calls += 1
        return self.answers.pop(0)

    def preview_retrieval(
        self, question: str, document_id: str | None = None
    ) -> tuple[list[RetrievedChunk], bool]:
        self.preview_calls += 1
        answer = self.answers.pop(0)
        return answer.used_chunks, not answer.abstained


ANSWERABLE = GoldenQuestion(
    question="What retriever does this paper use?",
    expected_answer="DPR",
    expected_section="2.2 Retriever: DPR",
)
NOT_IN_SOURCE = GoldenQuestion(
    question="How does it compare to RAPTOR?",
    expected_answer="Not answerable",
    is_abstain=True,
)


class TestRunner:
    def test_scores_a_correct_answer_and_a_correct_abstain(self):
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"]), abstained()])

        report = run_evaluation(engine, [ANSWERABLE, NOT_IN_SOURCE], judge=None)

        assert report.abstain_accuracy == 1.0
        assert report.recall_at_k == 1.0
        assert report.mrr == 1.0
        assert report.total_questions == 2

    def test_penalises_abstaining_on_an_answerable_question(self):
        engine = FakeEngine([abstained(["1 Introduction"]), abstained()])

        report = run_evaluation(engine, [ANSWERABLE, NOT_IN_SOURCE], judge=None)

        assert report.abstain_accuracy == 0.5

    def test_mrr_reflects_where_the_expected_section_ranked(self):
        engine = FakeEngine([answered("DPR", ["1 Introduction", "2.2 Retriever: DPR"])])

        report = run_evaluation(engine, [ANSWERABLE], judge=None)

        assert report.mrr == 0.5
        assert report.results[0].hit_rank == 2

    def test_captures_the_top_score_that_the_threshold_compared_against(self):
        # The threshold in engine.py compares against used_chunks[0].effective_score;
        # score=0.1 here stands in for a rejected-by-threshold rerank score.
        engine = FakeEngine([abstained(["2.2 Retriever: DPR"])])

        report = run_evaluation(engine, [ANSWERABLE], judge=None)

        assert report.results[0].top_score == 0.1

    def test_top_score_is_none_when_nothing_was_retrieved_at_all(self):
        engine = FakeEngine([abstained()])

        report = run_evaluation(engine, [NOT_IN_SOURCE], judge=None)

        assert report.results[0].top_score is None

    def test_retrieval_only_never_calls_answer_or_the_judge(self):
        # The whole point: ablation runs should not pay for LLM generation.
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"]), abstained()])
        judge_calls = []

        def judge(answer: str, context: str) -> bool:
            judge_calls.append(answer)
            return True

        report = run_evaluation(
            engine, [ANSWERABLE, NOT_IN_SOURCE], judge=judge, retrieval_only=True
        )

        assert engine.answer_calls == 0
        assert engine.preview_calls == 2
        assert judge_calls == []
        assert report.faithfulness is None

    def test_retrieval_only_still_scores_recall_mrr_and_abstain_accuracy(self):
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"]), abstained()])

        report = run_evaluation(
            engine, [ANSWERABLE, NOT_IN_SOURCE], judge=None, retrieval_only=True
        )

        assert report.recall_at_k == 1.0
        assert report.mrr == 1.0
        assert report.abstain_accuracy == 1.0

    def test_retrieval_metrics_ignore_abstain_questions(self):
        # An abstain question has no correct section, so it must not drag recall down.
        engine = FakeEngine([abstained()])

        report = run_evaluation(engine, [NOT_IN_SOURCE], judge=None)

        assert report.recall_at_k == 0.0
        assert report.abstain_accuracy == 1.0

    def test_faithfulness_is_none_when_the_judge_is_disabled(self):
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"])])

        report = run_evaluation(engine, [ANSWERABLE], judge=None)

        assert report.faithfulness is None
        assert report.results[0].faithful is None

    def test_faithfulness_only_judges_questions_that_produced_an_answer(self):
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"]), abstained()])
        judged = []

        def judge(answer: str, context: str) -> bool:
            judged.append(answer)
            return True

        report = run_evaluation(engine, [ANSWERABLE, NOT_IN_SOURCE], judge=judge)

        assert judged == ["DPR"]  # the abstain was never sent to the judge
        assert report.faithfulness == 1.0


class FakeJudgeLlm:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self.verdict


class TestLlmJudge:
    def test_accepts_a_yes_verdict(self):
        assert LlmJudge(FakeJudgeLlm("YES"))("answer", "context")

    def test_tolerates_casing_and_whitespace(self):
        assert LlmJudge(FakeJudgeLlm("  yes\n"))("answer", "context")

    def test_rejects_a_no_verdict(self):
        assert not LlmJudge(FakeJudgeLlm("NO"))("answer", "context")

    def test_treats_an_unparseable_verdict_as_unsupported(self):
        # A metric about trust must not pass on a verdict it could not read.
        assert not LlmJudge(FakeJudgeLlm("I'm not sure, possibly?"))("answer", "context")
