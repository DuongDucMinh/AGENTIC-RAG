"""Factories for dense and sparse embedding models."""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
# Load dense embedding model va cache lai.
def get_dense_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the dense embedding model used by Qdrant."""
    settings = get_settings()
    logger.info("Loading dense embedding model=%s", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model, model_kwargs={"device": "cpu"})


@lru_cache
# Load sparse embedding model khi bat Qdrant sparse mode.
def get_sparse_embeddings() -> FastEmbedSparse:
    """Load and cache the sparse BM25-style embedding model."""
    settings = get_settings()
    logger.info("Loading sparse embedding model=%s", settings.sparse_model)
    return FastEmbedSparse(model_name=settings.sparse_model)
