"""bge-m3 embeddings.

One model produces both vectors the hybrid retriever needs: a dense vector for
semantic similarity and lexical weights for keyword matching. CPU inference
measured ~40x slower than GPU for this model (see reranker for the matching
number), so it runs on CUDA despite the LLM sharing the same 6 GB - the two
never run at the same time within one request; see unload().
"""

from dataclasses import dataclass
from typing import Any

from abstractrag.core.config import EmbeddingSettings
from abstractrag.core.logging import get_logger
from abstractrag.rag.gpu import release_cuda_memory

logger = get_logger(__name__)


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: dict[int, float]  # token id -> weight


class BgeM3Embedder:
    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        # Imported and loaded on first use: it pulls in torch and costs seconds.
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            logger.info("loading %s on %s", self.settings.model, self.settings.device)
            self._model = BGEM3FlagModel(
                self.settings.model,
                devices=self.settings.device,
                use_fp16=self.settings.device != "cpu",
            )
        return self._model

    def embed(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []

        output = self.model.encode(
            texts,
            batch_size=self.settings.batch_size,
            max_length=self.settings.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"]
        sparse_weights = output["lexical_weights"]

        return [
            Embedding(
                dense=[float(value) for value in dense_vectors[i]],
                sparse={int(token): float(weight) for token, weight in sparse_weights[i].items()},
            )
            for i in range(len(texts))
        ]

    def embed_one(self, text: str) -> Embedding:
        return self.embed([text])[0]

    def unload(self) -> None:
        """Drop the model so the LLM has the GPU to itself. Reloads lazily
        (~15-20s) on next use - only worth calling right before an LLM call.

        A no-op on CPU: there is no GPU contention to relieve, so dropping and
        reloading would only pay the reload cost for nothing - measured as a
        real, repeated cost in the Docker deployment (embedding/reranker
        forced to CPU), where every agent retry's LLM call was reloading both
        models for no benefit."""
        if self._model is not None and self.settings.device != "cpu":
            self._model = None
            release_cuda_memory()
