"""Stage 2 retrieval: cut the candidates down to what actually answers the query.

A cross-encoder reads query and passage together, so it sees their interaction
instead of comparing two independent vectors. That costs one forward pass per
candidate, which is why it only runs on the ~50 candidates stage 1 returned.
Scores are normalized to 0-1 so the abstain threshold means something.
"""

from typing import Any

from abstractrag.core.config import RerankerSettings
from abstractrag.core.logging import get_logger
from abstractrag.rag.gpu import release_cuda_memory
from abstractrag.rag.models import RetrievedChunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, settings: RerankerSettings) -> None:
        self.settings = settings
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from FlagEmbedding import FlagReranker

            logger.info("loading %s on %s", self.settings.model, self.settings.device)
            self._model = FlagReranker(
                self.settings.model,
                devices=self.settings.device,
                use_fp16=self.settings.device != "cpu",
            )
        return self._model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if not self.settings.enabled:
            return candidates[:top_k]

        scores = self.model.compute_score(
            [[query, candidate.chunk.text] for candidate in candidates], normalize=True
        )
        if isinstance(scores, float):
            scores = [scores]

        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)

        candidates.sort(key=lambda candidate: candidate.effective_score, reverse=True)
        return candidates[:top_k]

    def unload(self) -> None:
        """Drop the model so the LLM has the GPU to itself. Reloads lazily
        (~5-7s) on next use - only worth calling right before an LLM call."""
        if self._model is not None:
            self._model = None
            release_cuda_memory()
