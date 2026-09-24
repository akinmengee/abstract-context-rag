"""Checking each claim against the passage it cited.

The model citing "[2]" is not evidence that passage [2] says what the sentence
claims - that is the gap this closes. Every claim goes to the judge in one
batched call: an answer is a handful of sentences, and one call costs one wait
instead of one per sentence.

Anything the judge does not clearly clear is treated as unsupported. A mechanism
whose job is trust must not pass a verdict it could not read.
"""

from abstractrag.core.logging import get_logger
from abstractrag.rag.models import Citation, ClaimVerdict, RetrievedChunk, VerifiedClaim
from abstractrag.rag.verification.claims import Claim

logger = get_logger(__name__)

SYSTEM_PROMPT = """You check whether each claim is supported by the passage it cites.

Reply with one line per claim, in order, in exactly this format:
<claim number>|YES or NO|<short reason, only when NO>

YES means every part of the claim appears in the passages it cites.
NO means the claim states something those passages do not.
Judge only support. Ignore style, completeness, and whether the claim is useful."""

_MISSING_VERDICT = "the judge returned no readable verdict for this claim"


class ClaimVerifier:
    def __init__(self, llm, judge_max_tokens: int = 8192) -> None:
        self.llm = llm
        self.judge_max_tokens = judge_max_tokens

    def verify(
        self,
        claims: list[Claim],
        citations: list[Citation],
        chunks: list[RetrievedChunk],
    ) -> list[VerifiedClaim]:
        """Check an answer's claims against the chunks its citations point at."""
        return self.verify_passages(claims, _passages_by_marker(citations, chunks))

    def verify_passages(
        self, claims: list[Claim], passages: dict[int, str]
    ) -> list[VerifiedClaim]:
        """Judge claims against passages the caller already resolved by marker.

        A summary's marker is a whole section rather than one chunk, so the
        mapping differs but the judging must not: the batched call, the claim
        numbering and the rule that an unreadable verdict fails are shared.
        """
        if not claims:
            return []

        # Keep each claim's position in the full answer: uncited claims and ones
        # citing a passage that is not there never reach the judge, and the
        # numbering the judge replies with has to line up with the full list.
        checkable = [
            (position, claim)
            for position, claim in enumerate(claims, start=1)
            if _known_markers(claim, passages)
        ]

        verdicts = self._ask_judge(checkable, passages) if checkable else {}
        return [
            self._resolve(claim, passages, verdicts.get(position))
            for position, claim in enumerate(claims, start=1)
        ]

    def _ask_judge(
        self, numbered_claims: list[tuple[int, Claim]], passages: dict[int, str]
    ) -> dict[int, tuple[ClaimVerdict, str]]:
        """One call for all checkable claims; returns verdicts keyed by claim number."""
        prompt = _build_prompt(numbered_claims, passages)
        response = self.llm.complete(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=self.judge_max_tokens,
        )
        return _parse_verdicts(response)

    def _resolve(
        self,
        claim: Claim,
        passages: dict[int, str],
        verdict: tuple[ClaimVerdict, str] | None,
    ) -> VerifiedClaim:
        if not claim.markers:
            return VerifiedClaim(
                text=claim.text, markers=[], verdict=ClaimVerdict.UNCITED, reason=""
            )

        unknown = [marker for marker in claim.markers if marker not in passages]
        if unknown:
            # The answer pointed at a source that was never in its context.
            markers = ", ".join(f"[{marker}]" for marker in unknown)
            return VerifiedClaim(
                text=claim.text,
                markers=claim.markers,
                verdict=ClaimVerdict.UNSUPPORTED,
                reason=f"cited {markers}, which is not among the retrieved passages",
            )

        if verdict is None:
            logger.warning("no verdict parsed for claim: %s", claim.text[:60])
            return VerifiedClaim(
                text=claim.text,
                markers=claim.markers,
                verdict=ClaimVerdict.UNSUPPORTED,
                reason=_MISSING_VERDICT,
            )

        decision, reason = verdict
        return VerifiedClaim(
            text=claim.text, markers=claim.markers, verdict=decision, reason=reason
        )


def _passages_by_marker(
    citations: list[Citation], chunks: list[RetrievedChunk]
) -> dict[int, str]:
    """Marker number -> the chunk text it refers to.

    Joined on chunk_id rather than list position, so this keeps working if
    build_context() ever changes how it orders or numbers the context blocks.
    """
    texts = {chunk.chunk.chunk_id: chunk.chunk.text for chunk in chunks}
    return {
        citation.marker: texts[citation.chunk_id]
        for citation in citations
        if citation.chunk_id in texts
    }


def _known_markers(claim: Claim, passages: dict[int, str]) -> bool:
    """Only claims whose every citation resolves are worth a judge call."""
    return bool(claim.markers) and all(marker in passages for marker in claim.markers)


def _build_prompt(numbered_claims: list[tuple[int, Claim]], passages: dict[int, str]) -> str:
    """Each cited passage listed once, then the claims that lean on them.

    Claim numbers are their position in the whole answer, so the judge's reply
    maps straight back even though uncited claims were left out of the prompt.
    """
    cited = sorted({marker for _, claim in numbered_claims for marker in claim.markers})
    passage_block = "\n\n".join(f"[{marker}] {passages[marker]}" for marker in cited)

    claim_block = "\n".join(
        f"{position}. (cites {' '.join(f'[{m}]' for m in claim.markers)}) {claim.text}"
        for position, claim in numbered_claims
    )
    return f"Passages:\n{passage_block}\n\nClaims:\n{claim_block}"


def _parse_verdicts(response: str) -> dict[int, tuple[ClaimVerdict, str]]:
    verdicts: dict[int, tuple[ClaimVerdict, str]] = {}
    for line in response.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue

        decision = parts[1].upper()
        if decision not in {"YES", "NO"}:
            continue  # unreadable verdict - left out so it resolves as unsupported

        reason = parts[2] if len(parts) > 2 else ""
        verdicts[int(parts[0])] = (
            ClaimVerdict.SUPPORTED if decision == "YES" else ClaimVerdict.UNSUPPORTED,
            "" if decision == "YES" else reason,
        )
    return verdicts
