"""Typed settings loaded from backend/config.yaml, overridable via ACR_* env vars."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_CONFIG_FILE = BACKEND_ROOT / "config.yaml"


class LLMSettings(BaseModel):
    base_url: str = "http://localhost:8080/v1"
    model: str = "qwen3-4b"
    ctx_size: int = 4096
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: int = 180


class EmbeddingSettings(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 8
    max_length: int = 1024
    dense_size: int = 1024


class RerankerSettings(BaseModel):
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    enabled: bool = True


class VerificationSettings(BaseModel):
    # One extra LLM call per answered question. On by default because flagging
    # unsupported claims is product behaviour, not a debug aid; turn it off with
    # ACR_VERIFICATION__ENABLED=false while iterating on anything else.
    enabled: bool = True
    # One verdict line per claim, and a summary can have many claims - so the
    # judge gets its own budget instead of raising llm.max_tokens for every call.
    judge_max_tokens: int = 2048
    # Passages + claims per judge call. A summary cites whole sections, and one
    # call carrying all of them (measured: most of a paper, ~16k tokens) was cut
    # by Ollama and the format instructions lost - 0/22 readable verdicts. At
    # ~4 chars/token this leaves room in llm.ctx_size (8192) for the reply.
    max_prompt_chars: int = 20000


class SummarizationSettings(BaseModel):
    # One LLM call per group, so this is the cost dial: bigger groups mean fewer
    # calls. Kept well under llm.ctx_size (6000 chars is roughly 1500 tokens) so
    # the prompt and the summary still fit without raising the context window.
    max_group_chars: int = 6000


class AgentSettings(BaseModel):
    # off: one retrieval pass, as in phases 1-3. corrective: an LLM grades the
    # retrieved chunks and rewrites the query when none answer it. multi_hop:
    # corrective, plus follow-up searches for what the first pass was missing.
    mode: str = Field(default="off", pattern="^(off|corrective|multi_hop)$")
    # Every rewrite or search is one more LLM call (~20-60s on the 4B model).
    max_rewrites: int = 1
    max_searches: int = 3
    # What several searches pile up must still fit llm.ctx_size with the prompt.
    max_context_chunks: int = 8


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "documents"


class ChunkingSettings(BaseModel):
    target_chars: int = 1800
    overlap_chars: int = 200
    min_chunk_chars: int = 200


class RetrievalSettings(BaseModel):
    mode: str = Field(default="hybrid", pattern="^(dense|sparse|hybrid)$")
    candidates: int = 50
    context_size: int = 5
    rrf_k: int = 60
    score_threshold: float = 0.3


class IngestionSettings(BaseModel):
    storage_dir: str = "data/documents"
    user_agent: str = "abstract-context-rag/0.1"

    def resolved_storage_dir(self) -> Path:
        """Absolute storage path; relative values are anchored at the repo root."""
        path = Path(self.storage_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ACR_",
        env_nested_delimiter="__",
        yaml_file=os.getenv("ACR_CONFIG_FILE", str(DEFAULT_CONFIG_FILE)),
        extra="ignore",
    )

    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    verification: VerificationSettings = VerificationSettings()
    summarization: SummarizationSettings = SummarizationSettings()
    agent: AgentSettings = AgentSettings()
    qdrant: QdrantSettings = QdrantSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    ingestion: IngestionSettings = IngestionSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Env wins over YAML so Docker and CI can override without editing files.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
