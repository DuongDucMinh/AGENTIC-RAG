"""Factories for dense and sparse embedding models."""

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

from app.core.config import get_settings

logger = logging.getLogger(__name__)

E5_PREFIX = "intfloat/multilingual-e5"


class PrefixedEmbeddings(Embeddings):
    """Apply model-specific text prefixes before delegating to a base embedder."""

    def __init__(self, base: HuggingFaceEmbeddings, query_prefix: str, document_prefix: str) -> None:
        self._base = base
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix

    @staticmethod
    def _prefix_text(prefix: str, text: str) -> str:
        stripped = text.strip()
        return stripped if stripped.startswith(prefix) else f"{prefix}{stripped}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_documents([self._prefix_text(self._document_prefix, text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(self._prefix_text(self._query_prefix, text))


@lru_cache
# Load dense embedding model va cache lai.
def get_dense_embeddings() -> Embeddings:
    """Load and cache the dense embedding model used by Qdrant."""
    settings = get_settings()
    model_name = settings.embedding_model
    logger.info("Loading dense embedding model=%s", model_name)

    base = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    if model_name.startswith(E5_PREFIX):
        logger.info("Applying E5 query/passage prefixes for model=%s", model_name)
        return PrefixedEmbeddings(base=base, query_prefix="query: ", document_prefix="passage: ")
    return base


@lru_cache
# Load sparse embedding model khi bat Qdrant sparse mode.
def get_sparse_embeddings() -> FastEmbedSparse:
    """Load and cache the sparse BM25-style embedding model."""
    settings = get_settings()
    logger.info("Loading sparse embedding model=%s", settings.sparse_model)
    return FastEmbedSparse(model_name=settings.sparse_model)
