"""Corrective and multi-hop agents, driven by a scripted LLM and a fake search."""

from abstractrag.core.config import AgentSettings
from abstractrag.rag.agents.base import ContextSelection
from abstractrag.rag.agents.corrective import CorrectiveAgent
from abstractrag.rag.agents.multi_hop import MultiHopAgent, _merge
from abstractrag.rag.agents.prompts import parse_grade, parse_plan, parse_rewrite
from abstractrag.rag.models import AgentStep, RetrievedChunk
from tests.conftest import make_chunk

THRESHOLD = 0.12


def chunk(text: str, index: int, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(text, index=index), score=score, rerank_score=score)


class ScriptedLlm:
    """Replies in order; records every prompt so tests can see what was asked."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


class FakeSearch:
    """Returns the chunks registered for a query (empty for anything else)."""

    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self.results = results
        self.queries: list[tuple[str, str | None]] = []

    def __call__(self, query: str, document_id: str | None) -> list[RetrievedChunk]:
        self.queries.append((query, document_id))
        return list(self.results.get(query, []))


class GpuSpy:
    def __init__(self) -> None:
        self.releases = 0

    def __call__(self) -> None:
        self.releases += 1


def corrective(llm, filter_chunks: bool = False, **settings) -> CorrectiveAgent:
    return CorrectiveAgent(
        llm=llm,
        settings=AgentSettings(**settings),
        score_threshold=THRESHOLD,
        filter_chunks=filter_chunks,
    )


class TestParsers:
    def test_grade_reads_passage_numbers(self):
        assert parse_grade("1, 3", count=5) == {1, 3}

    def test_grade_ignores_numbers_outside_the_passage_range(self):
        # "2018" in a chatty reply is not passage 2018.
        assert parse_grade("Passage 2 (from 2018) helps", count=5) == {2}

    def test_grade_none_means_nothing_kept(self):
        assert parse_grade("NONE", count=5) == set()

    def test_grade_that_cannot_be_read_says_so(self):
        assert parse_grade("I am not sure.", count=5) is None

    def test_rewrite_strips_labels_and_quotes(self):
        assert parse_rewrite('Query: "FEVER fact verification dataset"') == (
            "FEVER fact verification dataset"
        )

    def test_rewrite_takes_the_first_non_empty_line(self):
        assert parse_rewrite("\n  DPR training negatives\nextra") == "DPR training negatives"

    def test_plan_search_line_gives_the_query(self):
        assert parse_plan("SEARCH: DPR in-batch negatives") == "DPR in-batch negatives"

    def test_plan_with_two_searches_on_one_line_takes_the_first(self):
        reply = "SEARCH: which library does DPR use SEARCH: which library does ColBERT use"

        assert parse_plan(reply) == "which library does DPR use"

    def test_plan_done_gives_none(self):
        assert parse_plan("DONE") is None

    def test_plan_that_cannot_be_read_stops_searching(self):
        # Stopping early still answers from what was found; guessing a query does not.
        assert parse_plan("Maybe look at the training section?") is None


class TestCorrectiveAgent:
    def test_as_a_gate_it_keeps_the_whole_context_once_anything_answers(self):
        # Filtering dropped one side of a comparison in the real ablation run.
        chunks = [chunk("a", 0), chunk("b", 1), chunk("c", 2)]
        agent = corrective(ScriptedLlm(["1, 3"]))

        selection = agent.select("q", None, FakeSearch({"q": chunks}), GpuSpy())

        assert selection.sufficient
        assert [c.chunk.text for c in selection.chunks] == ["a", "b", "c"]
        assert selection.steps == [AgentStep(query="q", retrieved=3, kept=2)]

    def test_as_a_filter_it_keeps_only_the_chunks_the_grader_chose(self):
        chunks = [chunk("a", 0), chunk("b", 1), chunk("c", 2)]
        agent = corrective(ScriptedLlm(["1, 3"]), filter_chunks=True)

        selection = agent.select("q", None, FakeSearch({"q": chunks}), GpuSpy())

        assert [c.chunk.text for c in selection.chunks] == ["a", "c"]

    def test_rewrites_and_retries_when_the_grader_keeps_nothing(self):
        # The RAPTOR case: high reranker score, wrong system - graded out, then retried.
        search = FakeSearch({"q": [chunk("wrong system", 0)], "better q": [chunk("right", 1)]})
        agent = corrective(ScriptedLlm(["NONE", "better q", "1"]))

        selection = agent.select("q", "doc-1", search, GpuSpy())

        assert selection.sufficient
        assert [c.chunk.text for c in selection.chunks] == ["right"]
        assert search.queries == [("q", "doc-1"), ("better q", "doc-1")]

    def test_below_the_threshold_the_grader_is_skipped_and_the_query_rewritten(self):
        # The wrongly-abstained case: nothing worth grading, so go straight to a rewrite.
        search = FakeSearch({"q": [chunk("weak", 0, score=0.05)], "q2": [chunk("good", 1)]})
        llm = ScriptedLlm(["q2", "1"])

        selection = corrective(llm).select("q", None, search, GpuSpy())

        assert selection.sufficient
        assert len(llm.prompts) == 2  # rewrite + one grade, no grade for the weak search

    def test_gives_up_after_the_allowed_rewrites(self):
        search = FakeSearch({"q": [chunk("x", 0)], "q2": [chunk("y", 1)]})
        agent = corrective(ScriptedLlm(["NONE", "q2", "NONE"]), max_rewrites=1)

        selection = agent.select("q", None, search, GpuSpy())

        assert not selection.sufficient
        assert [step.query for step in selection.steps] == ["q", "q2"]

    def test_a_rewrite_identical_to_the_query_is_not_searched_again(self):
        search = FakeSearch({"q": [chunk("x", 0)]})

        selection = corrective(ScriptedLlm(["NONE", "Q"])).select("q", None, search, GpuSpy())

        assert not selection.sufficient
        assert len(search.queries) == 1

    def test_an_unreadable_grade_keeps_every_chunk(self):
        # A filter must never fall below the plain pipeline it sits on top of.
        chunks = [chunk("a", 0), chunk("b", 1)]

        selection = corrective(ScriptedLlm(["hmm"])).select(
            "q", None, FakeSearch({"q": chunks}), GpuSpy()
        )

        assert selection.sufficient
        assert len(selection.chunks) == 2

    def test_the_gpu_is_released_before_every_llm_call(self):
        search = FakeSearch({"q": [chunk("x", 0)], "q2": [chunk("y", 1)]})
        gpu = GpuSpy()
        llm = ScriptedLlm(["NONE", "q2", "1"])

        corrective(llm).select("q", None, search, gpu)

        assert gpu.releases >= len(llm.prompts)


class TestMultiHopAgent:
    @staticmethod
    def agent(llm, **settings) -> MultiHopAgent:
        agent_settings = AgentSettings(**settings)
        return MultiHopAgent(
            llm=llm,
            settings=agent_settings,
            corrective=CorrectiveAgent(
                llm=llm, settings=agent_settings, score_threshold=THRESHOLD, filter_chunks=True
            ),
        )

    def test_stops_when_the_planner_says_done(self):
        llm = ScriptedLlm(["1", "DONE"])

        selection = self.agent(llm).select(
            "q", None, FakeSearch({"q": [chunk("a", 0)]}), GpuSpy()
        )

        assert [c.chunk.text for c in selection.chunks] == ["a"]
        assert len(selection.steps) == 1

    def test_a_follow_up_search_adds_the_second_hop(self):
        # RAG says "our retriever is DPR"; only a search for DPR finds how DPR was trained.
        search = FakeSearch(
            {"q": [chunk("RAG uses DPR", 0)], "DPR training": [chunk("in-batch negatives", 1)]}
        )
        llm = ScriptedLlm(["1", "SEARCH: DPR training", "1", "DONE"])

        selection = self.agent(llm).select("q", None, search, GpuSpy())

        assert [c.chunk.text for c in selection.chunks] == ["RAG uses DPR", "in-batch negatives"]
        assert [step.query for step in selection.steps] == ["q", "DPR training"]

    def test_the_first_follow_up_offers_no_done_option_later_ones_do(self):
        search = FakeSearch({"q": [chunk("a", 0)], "s2": [chunk("b", 1)]})
        llm = ScriptedLlm(["1", "SEARCH: s2", "1", "DONE"])

        self.agent(llm).select("q", None, search, GpuSpy())

        first_plan, second_plan = llm.prompts[1], llm.prompts[3]
        assert "DONE" not in first_plan
        assert "DONE" in second_plan

    def test_the_planner_sees_what_was_found_so_far(self):
        llm = ScriptedLlm(["1", "DONE"])

        self.agent(llm).select("q", None, FakeSearch({"q": [chunk("RAG uses DPR", 0)]}), GpuSpy())

        assert "RAG uses DPR" in llm.prompts[1]

    def test_the_planner_sees_a_rejected_first_search_not_a_blank_slate(self):
        # rag.md's diagnosed bug: a passage graded "does not answer the
        # compound question" was discarded entirely, leaving the planner
        # reasoning from "(none yet)" even when the bridge fact ("RAG uses
        # DPR") was sitting right there in what the first search retrieved.
        search = FakeSearch(
            {"q": [chunk("RAG uses DPR", 0)], "DPR training": [chunk("in-batch negatives", 1)]}
        )
        llm = ScriptedLlm(["NONE", "SEARCH: DPR training", "1", "DONE"])

        selection = self.agent(llm, max_rewrites=0).select("q", None, search, GpuSpy())

        assert "RAG uses DPR" in llm.prompts[1]
        assert [c.chunk.text for c in selection.chunks] == ["in-batch negatives"]

    def test_stops_at_the_search_budget(self):
        search = FakeSearch({"q": [chunk("a", 0)], "s2": [chunk("b", 1)]})
        llm = ScriptedLlm(["1", "SEARCH: s2", "1"])

        selection = self.agent(llm, max_searches=2).select("q", None, search, GpuSpy())

        assert len(selection.steps) == 2
        assert llm.replies == []  # no third planning call was made

    def test_repeating_a_search_ends_the_loop(self):
        llm = ScriptedLlm(["1", "SEARCH: q"])

        selection = self.agent(llm).select(
            "q", None, FakeSearch({"q": [chunk("a", 0)]}), GpuSpy()
        )

        assert len(selection.steps) == 1

    def test_a_first_miss_can_still_be_rescued_by_a_later_hop(self):
        search = FakeSearch({"s2": [chunk("found later", 1)]})
        # first search empty -> rewrite "q-rw" also empty -> plan -> s2 graded
        llm = ScriptedLlm(["q-rw", "SEARCH: s2", "1", "DONE"])

        selection = self.agent(llm).select("q", None, search, GpuSpy())

        assert selection.sufficient
        assert [c.chunk.text for c in selection.chunks] == ["found later"]

    def test_nothing_found_anywhere_is_insufficient(self):
        llm = ScriptedLlm(["q-rw", "DONE"])

        selection = self.agent(llm).select("q", None, FakeSearch({}), GpuSpy())

        assert not selection.sufficient


class TestMerge:
    def test_drops_chunks_already_in_the_pool(self):
        a = chunk("a", 0)

        assert len(_merge([a], [chunk("a", 0), chunk("b", 1)], limit=8)) == 2

    def test_respects_the_context_budget_in_hop_order(self):
        pool = [chunk("a", 0), chunk("b", 1)]

        merged = _merge(pool, [chunk("c", 2), chunk("d", 3)], limit=3)

        assert [c.chunk.text for c in merged] == ["a", "b", "c"]


class FakeAgent:
    def __init__(self, selection: ContextSelection) -> None:
        self.selection = selection
        self.calls = 0

    def select(self, question, document_id, search, release_gpu) -> ContextSelection:
        self.calls += 1
        return self.selection


def test_selection_types_are_shared_with_the_engine():
    # The engine abstains or answers from whatever the agent returns.
    from tests.unit.test_engine import build_engine

    steps = [AgentStep(query="q", retrieved=5, kept=1)]
    engine = build_engine([], [], "Answer [1].")
    engine.agent = FakeAgent(ContextSelection([chunk("kept", 0)], True, steps))

    answer = engine.answer("q")

    assert engine.agent.calls == 1
    assert answer.agent_steps == steps
    assert [c.chunk.text for c in answer.used_chunks] == ["kept"]


def test_an_insufficient_agent_selection_abstains_and_keeps_its_trace():
    from tests.unit.test_engine import build_engine

    steps = [AgentStep(query="q", retrieved=5, kept=0)]
    engine = build_engine([], [], "unused")
    engine.agent = FakeAgent(ContextSelection([], False, steps))

    answer = engine.answer("q")

    assert answer.abstained
    assert answer.agent_steps == steps
