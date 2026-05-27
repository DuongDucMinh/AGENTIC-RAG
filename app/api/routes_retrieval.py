"""Retrieval debugging endpoint for hybrid search and reranking."""

import logging

from fastapi import APIRouter

from app.core.errors import RetrievalError
from app.retrieval.retriever import get_legal_retriever
from app.schemas.retrieval import RetrievalSearchRequest, RetrievalSearchResponse

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
logger = logging.getLogger(__name__)


@router.post("/search", response_model=RetrievalSearchResponse)
# Chay retrieval rieng de debug dense/BM25/RRF/rerank.
def search(request: RetrievalSearchRequest):
    """Search indexed legal chunks and return parent contexts with citations."""
    logger.info("Retrieval API requested query_len=%s top_k=%s", len(request.query), request.top_k)
    try:
        result = get_legal_retriever().search(request.query, top_k=request.top_k)
    except Exception as exc:
        logger.exception("Retrieval API failed")
        raise RetrievalError(f"Retrieval failed: {type(exc).__name__}") from exc
    return {
        "query": request.query,
        "contexts": result["contexts"],
        "citations": result["citations"],
        "trace": result["trace"] if request.debug else {},
    }
