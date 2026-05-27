"""Cross-encoder reranking for hybrid search candidates."""

import logging
import time
from functools import lru_cache

from langchain_core.documents import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_RERANKER_AVAILABLE = True


@lru_cache
def _cuda_available() -> bool:
    """Return True when a CUDA GPU is available for reranking."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        logger.exception("Failed to check CUDA availability for reranker")
        return False


@lru_cache
# Load cross-encoder reranker va cache model.
def _load_cross_encoder():
    """Load and cache the configured sentence-transformers CrossEncoder."""
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    logger.info("Loading reranker model=%s", settings.reranker_model)
    return CrossEncoder(settings.reranker_model, device="cuda")


# Rerank candidate chunks theo query va tra ve top_k.
def rerank(query: str, docs: list[Document], top_k: int) -> tuple[list[Document], list[float]]:
    """Rerank retrieved child chunks and return top documents with scores."""
    global _RERANKER_AVAILABLE
    settings = get_settings()
    gpu_available = _cuda_available()
    if not settings.enable_reranking or not docs or not _RERANKER_AVAILABLE or not gpu_available:
        logger.info(
            "Reranking skipped enabled=%s available=%s gpu_available=%s docs=%s",
            settings.enable_reranking,
            _RERANKER_AVAILABLE,
            gpu_available,
            len(docs),
        )
        return docs[:top_k], []

    try:
        start = time.perf_counter()
        model = _load_cross_encoder()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = [float(score) for score in model.predict(pairs)]
        ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Reranked documents input=%s output=%s latency_ms=%.2f", len(docs), min(top_k, len(docs)), elapsed_ms)
        return [doc for doc, _ in ranked[:top_k]], [score for _, score in ranked[:top_k]]
    except Exception:
        _RERANKER_AVAILABLE = False
        logger.exception("Reranker failed and has been disabled for this process")
        return docs[:top_k], []
