"""Centralized application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration shared across API, ingestion, retrieval, and agent code."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Prioritize project .env over ambient OS environment variables.
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    app_name: str = "Vietnamese Tax Legal RAG"
    app_env: str = "local"
    log_level: str = "INFO"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_rewrite_model: str = "llama-3.1-8b-instant"
    groq_judge_model: str = ""
    groq_answer_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "legal_tax_child_chunks"
    use_qdrant_sparse: bool = False

    hf_dataset_name: str = "th1nhng0/vietnamese-legal-documents"
    max_documents_to_index: int = 3000

    embedding_model: str = "intfloat/multilingual-e5-small"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    enable_reranking: bool = True
    retrieval_dense_top_k: int = 20
    retrieval_bm25_top_k: int = 20
    retrieval_fusion_top_k: int = 12
    retrieval_rerank_top_k: int = 5
    retrieval_rrf_k: int = 60
    retrieval_bm25_weight: int = 2
    retrieval_skip_rerank_below_docs: int = 2000

    child_chunk_size: int = 700
    child_chunk_overlap: int = 100
    fallback_parent_chars: int = 3500
    min_child_chars: int = 120

    parent_store_dir: Path = Field(default=Path("data/parent_store"))
    bm25_store_path: Path = Field(default=Path("data/bm25_index.pkl"))

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "vietnamese-tax-legal-rag"
    langsmith_endpoint: str = ""


@lru_cache
# Lay settings da cache va tao cac thu muc data can thiet.
def get_settings() -> Settings:
    """Return cached settings and ensure local data directories exist."""
    settings = Settings()
    settings.parent_store_dir.mkdir(parents=True, exist_ok=True)
    settings.bm25_store_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
