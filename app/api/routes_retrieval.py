"""Retrieval debugging endpoint for hybrid search and reranking."""

import logging

from fastapi import APIRouter

from app.retrieval.retriever import LegalRetriever
from app.schemas.retrieval import RetrievalSearchRequest, RetrievalSearchResponse

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
logger = logging.getLogger(__name__)


@router.post("/search", response_model=RetrievalSearchResponse)
# Chay retrieval rieng de debug dense/BM25/RRF/rerank.
def search(request: RetrievalSearchRequest):
    """Search indexed legal chunks and return parent contexts with citations."""
    logger.info("Retrieval API requested query_len=%s top_k=%s", len(request.query), request.top_k)
    result = LegalRetriever().search(request.query, top_k=request.top_k)
    return {
        "query": request.query,
        "contexts": result["contexts"],
        "citations": result["citations"],
        "trace": result["trace"] if request.debug else {},
    }
