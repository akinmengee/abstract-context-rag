"""Citation verification: splitting an answer into claims, then judging each one."""

from abstractrag.core.config import VerificationSettings
from abstractrag.rag.models import Answer, Citation, ClaimVerdict, RetrievedChunk, VerifiedClaim
from abstractrag.rag.verification.claims import Claim, split_claims
from abstractrag.rag.verification.verifier import ClaimVerifier
from tests.conftest import make_chunk


class TestVerdictModels:
    def test_answer_defaults_to_no_verified_claims(self):
        # Verification is optional; an unverified Answer must still be valid.
        assert Answer(text="x").verified_claims == []

    def test_a_verified_claim_defaults_to_an_empty_reason(self):
        # Only failures need explaining; supported claims carry no reason.
        claim = VerifiedClaim(text="x [1].", markers=[1], verdict=ClaimVerdict.SUPPORTED)

        assert claim.reason == ""


class TestSplitClaims:
    def test_each_sentence_becomes_a_claim_with_its_markers(self):
        claims = split_claims("DPR is the retriever [1][3]. It uses BERT [2].")

        assert [claim.text for claim in claims] == [
            "DPR is the retriever [1][3].",
            "It uses BERT [2].",
        ]
        assert [claim.markers for claim in claims] == [[1, 3], [2]]

    def test_sentence_without_a_marker_has_no_markers(self):
        assert split_claims("This is widely used.")[0].markers == []

    def test_decimal_numbers_do_not_split_a_sentence(self):
        # "0.88" has no whitespace after the period, so the sentence stays whole.
        assert len(split_claims("Recall was 0.88 on the test set [1].")) == 1

    def test_empty_answer_yields_no_claims(self):
        assert split_claims("   ") == []

    def test_duplicate_markers_in_one_sentence_are_collapsed(self):
        assert split_claims("Both parts come from [2] and [2].")[0].markers == [2]

    def test_a_final_sentence_without_punctuation_is_still_a_claim(self):
        claims = split_claims("First sentence [1]. Trailing one without a period [2]")

        assert len(claims) == 2
        assert claims[1].markers == [2]

    def test_claims_compare_by_value(self):
        assert split_claims("a [1].") == [Claim(text="a [1].", markers=[1])]


class FakeLlm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.last_prompt = ""
        self.last_max_tokens = None

    def complete(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        self.calls += 1
        self.last_prompt = messages[-1]["content"]
        self.last_max_tokens = max_tokens
        return self.response


CLAIMS = [
    Claim(text="DPR is the retriever [1].", markers=[1]),
    Claim(text="It was trained with Adam [2].", markers=[2]),
]
CITATIONS = [
    Citation(marker=1, chunk_id="c1", title="Paper", origin="x"),
    Citation(marker=2, chunk_id="c2", title="Paper", origin="x"),
]
CHUNKS = [
    RetrievedChunk(chunk=make_chunk("the retriever is DPR", index=0), score=1.0),
    RetrievedChunk(chunk=make_chunk("training used SGD", index=1), score=1.0),
]
# make_chunk derives deterministic ids, so point the citations at the real ones.
CITATIONS[0].chunk_id = CHUNKS[0].chunk.chunk_id
CITATIONS[1].chunk_id = CHUNKS[1].chunk.chunk_id


class TestClaimVerifier:
    def test_parses_one_verdict_line_per_claim(self):
        verifier = ClaimVerifier(FakeLlm("1|YES|\n2|NO|the passage never mentions Adam"))

        results = verifier.verify(CLAIMS, CITATIONS, CHUNKS)

        assert [r.verdict for r in results] == [ClaimVerdict.SUPPORTED, ClaimVerdict.UNSUPPORTED]
        assert "Adam" in results[1].reason

    def test_verifies_every_claim_in_a_single_llm_call(self):
        llm = FakeLlm("1|YES|\n2|YES|")

        ClaimVerifier(llm).verify(CLAIMS, CITATIONS, CHUNKS)

        assert llm.calls == 1

    def test_sends_the_cited_passage_text_to_the_judge(self):
        llm = FakeLlm("1|YES|\n2|YES|")

        ClaimVerifier(llm).verify(CLAIMS, CITATIONS, CHUNKS)

        assert "the retriever is DPR" in llm.last_prompt
        assert "training used SGD" in llm.last_prompt

    def test_uncited_claims_are_flagged_without_asking_the_llm(self):
        llm = FakeLlm("")

        results = ClaimVerifier(llm).verify([Claim(text="No source here.", markers=[])], [], [])

        assert results[0].verdict is ClaimVerdict.UNCITED
        assert llm.calls == 0  # nothing to verify it against

    def test_a_missing_verdict_line_counts_as_unsupported(self):
        # A trust metric must not silently pass a verdict it could not read.
        verifier = ClaimVerifier(FakeLlm("1|YES|"))

        results = verifier.verify(CLAIMS, CITATIONS, CHUNKS)

        assert results[1].verdict is ClaimVerdict.UNSUPPORTED

    def test_an_unparseable_verdict_line_counts_as_unsupported(self):
        verifier = ClaimVerifier(FakeLlm("1|maybe?|\n2|YES|"))

        results = verifier.verify(CLAIMS, CITATIONS, CHUNKS)

        assert results[0].verdict is ClaimVerdict.UNSUPPORTED

    def test_a_citation_marker_that_does_not_exist_is_unsupported(self):
        verifier = ClaimVerifier(FakeLlm("1|YES|"))

        results = verifier.verify([Claim(text="Fabricated [9].", markers=[9])], CITATIONS, CHUNKS)

        assert results[0].verdict is ClaimVerdict.UNSUPPORTED
        assert "[9]" in results[0].reason

    def test_no_claims_means_no_llm_call(self):
        llm = FakeLlm("")

        assert ClaimVerifier(llm).verify([], CITATIONS, CHUNKS) == []
        assert llm.calls == 0

    def test_verdicts_line_up_when_an_uncited_claim_sits_between_cited_ones(self):
        # Uncited claims never reach the judge, so the numbering it replies with
        # must still refer to each claim's position in the whole answer.
        claims = [
            Claim(text="No source here.", markers=[]),
            Claim(text="DPR is the retriever [1].", markers=[1]),
        ]
        verifier = ClaimVerifier(FakeLlm("2|NO|the passage says something else"))

        results = verifier.verify(claims, CITATIONS, CHUNKS)

        assert results[0].verdict is ClaimVerdict.UNCITED
        assert results[1].verdict is ClaimVerdict.UNSUPPORTED
        assert "something else" in results[1].reason


class TestVerifyPassages:
    def test_claims_are_judged_against_passages_given_directly(self):
        # Summarisation resolves its own markers, so it needs a way in that does
        # not go through citations and retrieved chunks.
        verifier = ClaimVerifier(FakeLlm("1|YES|"))

        results = verifier.verify_passages(
            [Claim(text="DPR is the retriever [1].", markers=[1])],
            {1: "We use DPR as the retriever."},
        )

        assert [result.verdict for result in results] == [ClaimVerdict.SUPPORTED]

    def test_a_marker_with_no_passage_is_unsupported_not_skipped(self):
        verifier = ClaimVerifier(FakeLlm("1|YES|"))

        results = verifier.verify_passages(
            [Claim(text="Trained on 4096 TPUs [7].", markers=[7])], {1: "some text"}
        )

        assert results[0].verdict is ClaimVerdict.UNSUPPORTED

    def test_no_claims_means_no_judge_call(self):
        llm = FakeLlm("1|YES|")

        assert ClaimVerifier(llm).verify_passages([], {1: "text"}) == []
        assert llm.calls == 0


class TestJudgeMaxTokens:
    def test_the_judge_call_uses_the_configured_max_tokens(self):
        # Not the default on purpose: if __init__ ever dropped judge_max_tokens,
        # or _ask_judge hardcoded a number instead of reading
        # self.judge_max_tokens, this must fail with a non-default value.
        llm = FakeLlm("1|YES|")
        verifier = ClaimVerifier(llm, judge_max_tokens=3333)

        verifier.verify_passages([Claim(text="DPR is the retriever [1].", markers=[1])], {1: "x"})

        assert llm.last_max_tokens == 3333

    def test_judge_max_tokens_defaults_to_the_configured_value(self):
        llm = FakeLlm("1|YES|")
        verifier = ClaimVerifier(llm)  # no judge_max_tokens given

        verifier.verify_passages([Claim(text="DPR is the retriever [1].", markers=[1])], {1: "x"})

        assert llm.last_max_tokens == VerificationSettings().judge_max_tokens


class EchoJudge:
    """Answers YES for every claim number it is shown; records each prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, messages, max_tokens=None) -> str:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        claims = prompt.split("Claims:\n", 1)[1]
        numbers = [line.split(".", 1)[0] for line in claims.splitlines() if line.strip()]
        return "\n".join(f"{number}|YES|" for number in numbers)


class TestJudgeBatching:
    def test_claims_whose_passages_overflow_the_budget_go_to_separate_calls(self):
        # A summary cites whole sections; one prompt with all of them overflowed
        # the context window and lost the format instructions (0/22 verdicts).
        llm = EchoJudge()
        claims = [Claim(text=f"Claim {n} [{n}].", markers=[n]) for n in (1, 2, 3)]
        passages = {n: "x" * 400 for n in (1, 2, 3)}

        results = ClaimVerifier(llm, max_prompt_chars=900).verify_passages(claims, passages)

        assert len(llm.prompts) == 2
        assert all(len(prompt) < 1200 for prompt in llm.prompts)
        assert [r.verdict for r in results] == [ClaimVerdict.SUPPORTED] * 3

    def test_verdicts_from_later_calls_map_back_to_the_right_claims(self):
        llm = EchoJudge()
        claims = [
            Claim(text="Uncited.", markers=[]),
            Claim(text="A [1].", markers=[1]),
            Claim(text="B [2].", markers=[2]),
        ]

        results = ClaimVerifier(llm, max_prompt_chars=500).verify_passages(
            claims, {1: "y" * 400, 2: "z" * 400}
        )

        assert [r.verdict for r in results] == [
            ClaimVerdict.UNCITED,
            ClaimVerdict.SUPPORTED,
            ClaimVerdict.SUPPORTED,
        ]

    def test_a_small_answer_still_takes_one_call(self):
        llm = EchoJudge()
        claims = [Claim(text="A [1].", markers=[1]), Claim(text="B [2].", markers=[2])]

        ClaimVerifier(llm).verify_passages(claims, {1: "short", 2: "short"})

        assert len(llm.prompts) == 1

    def test_one_claim_citing_more_than_the_budget_has_its_passages_trimmed(self):
        llm = EchoJudge()
        claim = Claim(text="Everything [1][2].", markers=[1, 2])

        results = ClaimVerifier(llm, max_prompt_chars=1000).verify_passages(
            [claim], {1: "a" * 5000, 2: "b" * 5000}
        )

        assert len(llm.prompts[0]) < 1200
        assert results[0].verdict is ClaimVerdict.SUPPORTED
