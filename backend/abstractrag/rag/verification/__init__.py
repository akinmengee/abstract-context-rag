"""Citation verification: does the cited passage actually say this?

A model citing "[2]" is not evidence that passage [2] supports the sentence it
is attached to. This package closes that gap, and it is what makes the project's
grounding claim measurable rather than aspirational.

  claims.py    splits an answer into one claim per sentence, with its [n] markers
  verifier.py  checks every claim in one batched LLM call, flags what fails

Three verdicts: SUPPORTED, UNSUPPORTED (it cites something that does not say
this), UNCITED (asserted with no source at all). Unsupported claims are flagged,
never deleted or regenerated - see rag.md section 9, transparency over cleanup.

Runs on every answered question by default; `verification.enabled` turns it off.
Abstains are never verified: nothing was claimed.
"""
