"""LLM-as-judge checks: faithfulness and correctness.

Neither can be computed with arithmetic: deciding whether "DPR is trained
jointly with the generator" is supported by a passage, or matches a reference
answer, requires reading both. The same local model that answers also judges -
no cloud API, consistent with the rest of the project.

Each is one call per answered question, which is why the runner makes them optional.
"""

JUDGE_SYSTEM_PROMPT = """You check whether an answer is supported by its source \
context.

Reply with exactly one word: YES or NO.
- YES: every factual claim in the answer appears in the context.
- NO: the answer contains at least one claim the context does not support.

Judge only support, not style, completeness, or whether the answer is a good one."""

JUDGE_USER_TEMPLATE = """Context:
{context}

Answer:
{answer}

Is every claim in the answer supported by the context? Reply YES or NO."""


# A metric has to give the same verdict for the same answer: at the answering
# temperature, two near-identical answers were once judged differently.
JUDGE_TEMPERATURE = 0.0

CORRECTNESS_SYSTEM_PROMPT = """You check whether an answer to a question is correct, \
given a reference answer.

Reply with exactly one word: YES or NO.
- YES: the answer states the key facts of the reference answer. Different wording \
and extra correct detail are fine.
- NO: the answer misses a key fact of the reference, contradicts it, or does not \
answer the question."""

CORRECTNESS_USER_TEMPLATE = """Question:
{question}

Reference answer:
{expected}

Answer to check:
{answer}

Is the answer correct? Reply YES or NO."""


class LlmJudge:
    """Wraps an LLM client so the runner can call it as judge(answer, context)."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, answer: str, context: str) -> bool:
        verdict = self.llm.complete(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(context=context, answer=answer),
                },
            ],
            temperature=JUDGE_TEMPERATURE,
        )
        return _is_yes(verdict)


class CorrectnessJudge:
    """Did the answer get it right? Called as judge(question, expected, answer).

    Faithfulness cannot tell this apart: a multi-hop answer that never found the
    second hop can be perfectly faithful to the one passage it did find.
    """

    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, question: str, expected: str, answer: str) -> bool:
        verdict = self.llm.complete(
            [
                {"role": "system", "content": CORRECTNESS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CORRECTNESS_USER_TEMPLATE.format(
                        question=question, expected=expected, answer=answer
                    ),
                },
            ],
            temperature=JUDGE_TEMPERATURE,
        )
        return _is_yes(verdict)


def _is_yes(verdict: str) -> bool:
    # Anything that is not a clear YES counts as a fail: with a metric about
    # trust, an unparseable verdict should not quietly pass.
    return verdict.strip().upper().startswith("YES")
