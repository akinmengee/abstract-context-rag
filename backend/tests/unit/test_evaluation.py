"""Evaluation metrics, the golden set format, and the runner that turns it into a report."""

import pytest
from pydantic import ValidationError

from abstractrag.core.config import RetrievalSettings, Settings
from abstractrag.rag.evaluation.judge import CorrectnessJudge, LlmJudge
from abstractrag.rag.evaluation.metrics import (
    evidence_recall,
    first_hit_rank,
    recall_at_k,
    reciprocal_rank,
)
from abstractrag.rag.evaluation.models import Evidence, GoldenQuestion, RetrievedSection
from abstractrag.rag.evaluation.runner import (
    MissingPapersError,
    default_golden_sets,
    load_golden_set,
    run_evaluation,
    select_split,
)
from abstractrag.rag.models import Answer, RetrievedChunk, VerifiedClaim
from tests.conftest import make_chunk

RAG = "2005.11401"
DPR = "2004.04906"
RAG_ORIGIN = f"https://arxiv.org/abs/{RAG}"
DPR_ORIGIN = f"https://arxiv.org/abs/{DPR}"


def ref(section: str, origin: str = RAG_ORIGIN) -> RetrievedSection:
    return RetrievedSection(origin=origin, section=section)


RANKING = [ref("1 Introduction"), ref("2.2 Retriever: DPR"), ref("3 Results")]
RETRIEVER = Evidence(paper=RAG, section="2.2 Retriever: DPR")


class TestFirstHitRank:
    def test_returns_one_based_rank_of_the_expected_section(self):
        assert first_hit_rank(RANKING, RETRIEVER) == 2

    def test_returns_none_when_the_section_was_never_retrieved(self):
        assert first_hit_rank(RANKING, Evidence(paper=RAG, section="5 Conclusion")) is None

    def test_ignores_case_and_surrounding_whitespace(self):
        # Docling section titles vary in spacing; the golden set is hand-written.
        assert first_hit_rank(RANKING, Evidence(paper=RAG, section="  2.2 retriever: dpr ")) == 2

    def test_the_same_section_name_in_another_paper_is_not_a_hit(self):
        # Every paper has a "1 Introduction" - the paper is part of the key.
        assert first_hit_rank(RANKING, Evidence(paper=DPR, section="1 Introduction")) is None

    def test_returns_none_for_an_empty_ranking(self):
        assert first_hit_rank([], RETRIEVER) is None


class TestRecallAtK:
    def test_hits_when_the_section_is_within_k(self):
        assert recall_at_k(RANKING, RETRIEVER, k=2)

    def test_misses_when_the_section_ranks_below_k(self):
        assert not recall_at_k(RANKING, Evidence(paper=RAG, section="3 Results"), k=2)


class TestReciprocalRank:
    def test_is_one_over_the_first_hit_position(self):
        assert reciprocal_rank(RANKING, RETRIEVER) == 0.5

    def test_is_zero_when_the_section_was_never_retrieved(self):
        assert reciprocal_rank(RANKING, Evidence(paper=RAG, section="5 Conclusion")) == 0.0


class TestEvidenceRecall:
    TWO_HOPS = [RETRIEVER, Evidence(paper=DPR, section="3 Dense Passage Retriever")]

    def test_all_evidence_found(self):
        ranking = [ref("2.2 Retriever: DPR"), ref("3 Dense Passage Retriever", DPR_ORIGIN)]

        assert evidence_recall(ranking, self.TWO_HOPS) == 1.0

    def test_one_hop_found_gets_partial_credit(self):
        # Found the first hop, missed the second: a different failure from finding nothing.
        assert evidence_recall([ref("2.2 Retriever: DPR")], self.TWO_HOPS) == 0.5

    def test_nothing_found(self):
        assert evidence_recall([ref("1 Introduction")], self.TWO_HOPS) == 0.0


class TestGoldenQuestion:
    def test_a_single_question_needs_exactly_one_evidence(self):
        with pytest.raises(ValidationError):
            GoldenQuestion(question="q", expected_answer="a", kind="single", evidence=[])

    def test_a_multi_hop_question_needs_at_least_two_evidence(self):
        with pytest.raises(ValidationError):
            GoldenQuestion(
                question="q", expected_answer="a", kind="multi_hop", evidence=[RETRIEVER]
            )

    def test_an_abstain_question_has_no_evidence(self):
        with pytest.raises(ValidationError):
            GoldenQuestion(question="q", expected_answer="a", kind="abstain", evidence=[RETRIEVER])

    def test_a_global_question_needs_a_scope(self):
        # "Summarise this paper" means nothing without saying which paper.
        with pytest.raises(ValidationError):
            GoldenQuestion(question="Summarise this paper.", expected_answer="a", kind="global")

    def test_a_global_question_has_no_evidence(self):
        with pytest.raises(ValidationError):
            GoldenQuestion(
                question="q", expected_answer="a", kind="global", scope=RAG, evidence=[RETRIEVER]
            )

    def test_a_question_is_eval_unless_marked_otherwise(self):
        # Existing golden files carry no split: they must stay out of training.
        question = GoldenQuestion(question="q", expected_answer="a", kind="abstain")

        assert question.split == "eval"

    def test_an_unknown_split_is_rejected(self):
        with pytest.raises(ValidationError):
            GoldenQuestion(question="q", expected_answer="a", kind="abstain", split="test")

    @pytest.mark.parametrize("path", default_golden_sets(), ids=lambda path: path.name)
    def test_every_golden_file_loads(self, path):
        # A malformed golden file must fail here, not halfway through a slow run.
        assert load_golden_set(path)


class TestSelectSplit:
    QUESTIONS = [
        GoldenQuestion(question="e", expected_answer="a", kind="abstain"),
        GoldenQuestion(question="t", expected_answer="a", kind="abstain", split="train"),
    ]

    def test_eval_leaves_training_questions_out(self):
        assert [q.question for q in select_split(self.QUESTIONS, "eval")] == ["e"]

    def test_train_selects_only_training_questions(self):
        assert [q.question for q in select_split(self.QUESTIONS, "train")] == ["t"]

    def test_all_keeps_everything(self):
        assert len(select_split(self.QUESTIONS, "all")) == 2


def chunks_from(sections: list[tuple[str, str]], score: float) -> list[RetrievedChunk]:
    """(origin, section) pairs -> retrieved chunks, in rank order."""
    chunks = []
    for i, (origin, section) in enumerate(sections):
        chunk = make_chunk(f"chunk {i}", index=i, section=section)
        chunk.metadata.origin = origin
        chunks.append(RetrievedChunk(chunk=chunk, score=score))
    return chunks


def answered(text: str, sections: list[str], origin: str = RAG_ORIGIN) -> Answer:
    return Answer(text=text, used_chunks=chunks_from([(origin, s) for s in sections], 1.0))


def abstained(sections: list[str] | None = None) -> Answer:
    return Answer(
        text="This source does not contain that information.",
        abstained=True,
        used_chunks=chunks_from([(RAG_ORIGIN, s) for s in sections or []], 0.1),
    )


class FakeStore:
    def __init__(self, origins: list[str]) -> None:
        self.origins = origins

    def list_documents(self) -> list[dict]:
        return [{"document_id": f"doc-{origin}", "origin": origin} for origin in self.origins]


class FakeEngine:
    """Replays a canned Answer per question, in the order the runner asks.

    Both answer() and preview_retrieval() draw from the same canned list, so a
    test can build one set of expectations with answered()/abstained() and use
    it to drive either path - only call counts tell them apart.
    """

    def __init__(self, answers: list[Answer], origins: list[str] | None = None) -> None:
        self.answers = list(answers)
        self.settings = Settings(retrieval=RetrievalSettings(context_size=3))
        self.store = FakeStore(origins if origins is not None else [RAG_ORIGIN, DPR_ORIGIN])
        self.answer_calls = 0
        self.ask_calls = 0
        self.preview_calls = 0
        self.document_ids: list[str | None] = []

    def answer(self, question: str, document_id: str | None = None) -> Answer:
        self.answer_calls += 1
        self.document_ids.append(document_id)
        return self.answers.pop(0)

    def ask(self, question: str, document_id: str | None = None) -> Answer:
        self.ask_calls += 1
        self.document_ids.append(document_id)
        return self.answers.pop(0)

    def preview_retrieval(
        self, question: str, document_id: str | None = None
    ) -> tuple[list[RetrievedChunk], bool]:
        self.preview_calls += 1
        self.document_ids.append(document_id)
        answer = self.answers.pop(0)
        return answer.used_chunks, not answer.abstained


ANSWERABLE = GoldenQuestion(
    question="What retriever does this paper use?",
    expected_answer="DPR",
    kind="single",
    scope=RAG,
    evidence=[RETRIEVER],
)
NOT_IN_SOURCE = GoldenQuestion(
    question="How does it compare to RAPTOR?",
    expected_answer="Not answerable",
    kind="abstain",
    scope=RAG,
)
MULTI_HOP = GoldenQuestion(
    question="What was the retriever RAG uses trained on?",
    expected_answer="NQ, TriviaQA, WQ, TREC, SQuAD",
    kind="multi_hop",
    evidence=[RETRIEVER, Evidence(paper=DPR, section="5.1 Datasets")],
)


class TestRunner:
    def test_scores_a_correct_answer_and_a_correct_abstain(self):
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"]), abstained()])

        report = run_evaluation(engine, [ANSWERABLE, NOT_IN_SOURCE])

        assert report.abstain_accuracy == 1.0
        assert report.recall_at_k == 1.0
        assert report.mrr == 1.0
        assert report.total_questions == 2

    def test_a_scoped_question_is_asked_against_its_papers_document(self):
        # "this paper" means nothing across a corpus - the scope pins it down.
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"])])

        run_evaluation(engine, [ANSWERABLE])

        assert engine.document_ids == [f"doc-{RAG_ORIGIN}"]

    def test_an_unscoped_question_searches_the_whole_corpus(self):
        engine = FakeEngine([answered("x", ["2.2 Retriever: DPR"])])

        run_evaluation(engine, [MULTI_HOP])

        assert engine.document_ids == [None]

    def test_a_paper_that_is_not_ingested_stops_the_run_before_any_question(self):
        engine = FakeEngine([answered("x", [])], origins=[RAG_ORIGIN])

        with pytest.raises(MissingPapersError, match=DPR):
            run_evaluation(engine, [ANSWERABLE, MULTI_HOP])
        assert engine.answer_calls == 0

    def test_penalises_abstaining_on_an_answerable_question(self):
        engine = FakeEngine([abstained(["1 Introduction"]), abstained()])

        report = run_evaluation(engine, [ANSWERABLE, NOT_IN_SOURCE])

        assert report.abstain_accuracy == 0.5

    def test_mrr_reflects_where_the_expected_section_ranked(self):
        engine = FakeEngine([answered("DPR", ["1 Introduction", "2.2 Retriever: DPR"])])

        report = run_evaluation(engine, [ANSWERABLE])

        assert report.mrr == 0.5
        assert report.results[0].hit_rank == 2

    def test_a_multi_hop_question_scores_evidence_recall_not_recall_at_k(self):
        engine = FakeEngine(
            [answered("DPR", ["2.2 Retriever: DPR"]), answered("x", ["2.2 Retriever: DPR"])]
        )

        report = run_evaluation(engine, [ANSWERABLE, MULTI_HOP])

        assert report.results[1].evidence_recall == 0.5
        assert report.evidence_recall == 0.5
        assert report.recall_at_k == 1.0  # single questions only

    def test_scores_are_broken_down_by_kind(self):
        engine = FakeEngine(
            [answered("DPR", ["2.2 Retriever: DPR"]), abstained(), answered("x", [])]
        )

        report = run_evaluation(engine, [ANSWERABLE, NOT_IN_SOURCE, MULTI_HOP])

        assert set(report.by_kind) == {"single", "abstain", "multi_hop"}
        assert report.by_kind["multi_hop"].evidence_recall == 0.0
        assert report.by_kind["single"].evidence_recall is None

    def test_mean_context_chunks_counts_only_answered_questions(self):
        engine = FakeEngine([answered("DPR", ["a", "b", "c"]), abstained(["a"])])

        report = run_evaluation(engine, [ANSWERABLE, NOT_IN_SOURCE])

        assert report.mean_context_chunks == 3.0

    def test_captures_the_top_score_that_the_threshold_compared_against(self):
        engine = FakeEngine([abstained(["2.2 Retriever: DPR"])])

        report = run_evaluation(engine, [ANSWERABLE])

        assert report.results[0].top_score == 0.1

    def test_top_score_is_none_when_nothing_was_retrieved_at_all(self):
        engine = FakeEngine([abstained()])

        report = run_evaluation(engine, [NOT_IN_SOURCE])

        assert report.results[0].top_score is None

    def test_retrieval_only_never_calls_answer_or_either_judge(self):
        # The whole point: ablation runs should not pay for LLM generation.
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"]), abstained()])
        calls = []

        report = run_evaluation(
            engine,
            [ANSWERABLE, NOT_IN_SOURCE],
            judge=lambda answer, context: calls.append(answer) or True,
            correctness=lambda question, expected, answer: calls.append(answer) or True,
            retrieval_only=True,
        )

        assert engine.answer_calls == 0
        assert engine.preview_calls == 2
        assert calls == []
        assert report.faithfulness is None
        assert report.accuracy is None

    def test_retrieval_only_passes_the_scope_through(self):
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"])])

        run_evaluation(engine, [ANSWERABLE], retrieval_only=True)

        assert engine.document_ids == [f"doc-{RAG_ORIGIN}"]

    def test_faithfulness_only_judges_questions_that_produced_an_answer(self):
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"]), abstained()])
        judged = []

        report = run_evaluation(
            engine,
            [ANSWERABLE, NOT_IN_SOURCE],
            judge=lambda answer, context: judged.append(answer) or True,
        )

        assert judged == ["DPR"]
        assert report.faithfulness == 1.0


class TestCorrectness:
    def test_an_answered_question_is_sent_to_the_judge(self):
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"])])
        asked = []

        report = run_evaluation(
            engine,
            [ANSWERABLE],
            correctness=lambda q, expected, answer: asked.append((expected, answer)) or True,
        )

        assert asked == [("DPR", "DPR")]
        assert report.accuracy == 1.0

    def test_a_correct_abstain_is_correct_without_asking_the_judge(self):
        engine = FakeEngine([abstained()])
        asked = []

        report = run_evaluation(
            engine, [NOT_IN_SOURCE], correctness=lambda *args: asked.append(args) or False
        )

        assert asked == []
        assert report.results[0].correct is True

    def test_abstaining_on_an_answerable_question_is_incorrect(self):
        engine = FakeEngine([abstained()])

        report = run_evaluation(engine, [ANSWERABLE], correctness=lambda *args: True)

        assert report.results[0].correct is False

    def test_answering_an_abstain_question_is_incorrect(self):
        engine = FakeEngine([answered("made up", ["1 Introduction"])])

        report = run_evaluation(engine, [NOT_IN_SOURCE], correctness=lambda *args: True)

        assert report.results[0].correct is False

    def test_accuracy_is_none_when_the_judge_is_disabled(self):
        engine = FakeEngine([answered("DPR", ["2.2 Retriever: DPR"])])

        report = run_evaluation(engine, [ANSWERABLE])

        assert report.accuracy is None
        assert report.results[0].correct is None


GLOBAL = GoldenQuestion(
    question="Summarise this paper.", expected_answer="RAG", kind="global", scope=RAG
)


def summary(text: str, verdicts: list[str]) -> Answer:
    """A map-reduce style answer: no retrieved chunks, verified claims only."""
    return Answer(
        text=text,
        verified_claims=[
            VerifiedClaim(text=f"claim {i}", markers=[1], verdict=verdict)
            for i, verdict in enumerate(verdicts)
        ],
    )


class TestRoutedAndTimed:
    def test_entry_ask_goes_through_the_router(self):
        # Global questions only reach summarize() through ask().
        engine = FakeEngine([summary("RAG [1].", ["supported"])])

        run_evaluation(engine, [GLOBAL], entry="ask")

        assert engine.ask_calls == 1
        assert engine.answer_calls == 0

    def test_each_question_is_timed(self, monkeypatch):
        clock = iter([10.0, 12.5])
        monkeypatch.setattr(
            "abstractrag.rag.evaluation.runner.time.perf_counter", lambda: next(clock)
        )
        engine = FakeEngine([summary("RAG [1].", ["supported"])])

        report = run_evaluation(engine, [GLOBAL], entry="ask")

        assert report.results[0].seconds == 2.5
        assert report.by_kind["global"].mean_seconds == 2.5

    def test_supported_claims_is_the_share_verification_cleared(self):
        engine = FakeEngine(
            [summary("RAG [1].", ["supported", "supported", "unsupported", "uncited"])]
        )

        report = run_evaluation(engine, [GLOBAL], entry="ask")

        assert report.results[0].supported_claims == 0.5
        assert report.supported_claims == 0.5

    def test_an_answer_with_no_retrieved_context_is_not_sent_to_the_faithfulness_judge(self):
        # A summary has no used_chunks; judging it against empty context always says NO.
        engine = FakeEngine([summary("RAG [1].", ["supported"])])
        judged = []

        report = run_evaluation(
            engine,
            [GLOBAL],
            judge=lambda answer, context: judged.append(answer) or True,
            entry="ask",
        )

        assert judged == []
        assert report.results[0].faithful is None
        # Nothing was judged: 0.00 would read as "every answer unfaithful".
        assert report.faithfulness is None


class FakeJudgeLlm:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.prompts: list[str] = []

    def complete(self, messages: list[dict[str, str]], **options) -> str:
        self.prompts.append(messages[-1]["content"])
        self.options = options
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


class TestCorrectnessJudge:
    def test_accepts_a_yes_verdict(self):
        assert CorrectnessJudge(FakeJudgeLlm("YES"))("q", "expected", "answer")

    def test_treats_an_unparseable_verdict_as_incorrect(self):
        assert not CorrectnessJudge(FakeJudgeLlm("hmm"))("q", "expected", "answer")

    def test_judges_deterministically(self):
        # The same answer must always get the same verdict.
        llm = FakeJudgeLlm("YES")

        CorrectnessJudge(llm)("q", "expected", "answer")

        assert llm.options["temperature"] == 0.0

    def test_the_prompt_carries_question_reference_and_answer(self):
        llm = FakeJudgeLlm("YES")

        CorrectnessJudge(llm)("Which retriever?", "DPR", "It uses DPR.")

        assert all(part in llm.prompts[0] for part in ["Which retriever?", "DPR", "It uses DPR."])
