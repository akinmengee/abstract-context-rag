"""Prompts and context assembly for grounded answering."""

from abstractrag.rag.models import Citation, RetrievedChunk

# The model returns this exact token when the context cannot answer the question,
# which is cheaper and far more reliable than detecting a hedged answer afterwards.
ABSTAIN_SENTINEL = "NOT_IN_SOURCE"
ABSTAIN_MESSAGE = "This source does not contain that information."

SYSTEM_PROMPT = f"""You answer questions about a document using only the numbered \
context blocks provided.

Rules:
- Use only the context. Never add outside knowledge, even if you are confident.
- Cite the block number in square brackets after every claim, like [2]. Multiple \
blocks can support one claim: [1][3].
- Quote numbers, hyperparameters and results exactly as written instead of \
paraphrasing them.
- If the context does not answer the question, reply with exactly {ABSTAIN_SENTINEL} \
and nothing else.
- Be concise. Do not repeat the question or explain your reasoning."""

USER_TEMPLATE = """Context:
{context}

Question: {question}"""


def order_for_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Put the strongest chunks at the start and end of the context.

    Models recall the middle of a long context worst ("lost in the middle"), so the
    best-ranked chunk goes first, the second-best last, and so on inwards.
    """
    ranked = sorted(chunks, key=lambda chunk: chunk.effective_score, reverse=True)
    front: list[RetrievedChunk] = []
    back: list[RetrievedChunk] = []
    for position, chunk in enumerate(ranked):
        (front if position % 2 == 0 else back).append(chunk)
    return front + list(reversed(back))


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Render numbered context blocks and the citations those numbers refer to."""
    blocks: list[str] = []
    citations: list[Citation] = []

    for marker, retrieved in enumerate(chunks, start=1):
        metadata = retrieved.chunk.metadata
        location = " — ".join(
            part
            for part in (
                metadata.title,
                metadata.section,
                f"page {metadata.page}" if metadata.page else None,
            )
            if part
        )
        blocks.append(f"[{marker}] {location}\n{retrieved.chunk.text}")
        citations.append(
            Citation(
                marker=marker,
                chunk_id=retrieved.chunk.chunk_id,
                title=metadata.title,
                section=metadata.section,
                page=metadata.page,
                origin=metadata.origin,
            )
        )

    return "\n\n".join(blocks), citations


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]
