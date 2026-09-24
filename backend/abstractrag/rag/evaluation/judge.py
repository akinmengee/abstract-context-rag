"""LLM-as-judge faithfulness check.

Faithfulness is the one metric that cannot be computed with arithmetic: deciding
whether "DPR is trained jointly with the generator" is supported by a passage
requires reading both. The same local model that answers also judges - no cloud
API, consistent with the rest of the project.

One call per answered question, which is why the runner makes it optional.
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
            ]
        )
        # Anything that is not a clear YES counts as unsupported: with a metric
        # about trust, an unparseable verdict should not quietly pass.
        return verdict.strip().upper().startswith("YES")
