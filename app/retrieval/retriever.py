"""High-level retrieval pipeline: hybrid search, rerank, parent context load."""

import logging
from typing import Any

from langchain_core.documents import Document

from app.retrieval.parent_store import ParentStore
from app.retrieval.qdrant_store import get_vector_store
from app.retrieval.reranker import rerank
from app.retrieval.bm25_store import BM25Store
from app.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


# Chuyen metadata cua parent chunk thanh citation public.
def _citation_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert stored metadata into the public citation shape."""
    return {
        "title": metadata.get("title", ""),
        "so_ky_hieu": metadata.get("so_ky_hieu", ""),
        "article_number": metadata.get("article_number"),
        "article_title": metadata.get("article_title"),
        "status": metadata.get("tinh_trang_hieu_luc", ""),
        "doc_id": metadata.get("doc_id", ""),
        "parent_id": metadata.get("parent_id", ""),
    }


class LegalRetriever:
    """Retrieve legal contexts suitable for grounded answer generation."""

    # Khoi tao parent store va BM25 sidecar.
    def __init__(self) -> None:
        """Initialize access to the parent context store."""
        self.parent_store = ParentStore()
        self.bm25_store = BM25Store()

    # Chay dense search, BM25, RRF, rerank va load parent contexts.
    def search(self, query: str, top_k: int = 5, hybrid_k: int = 50, rerank_k: int = 10) -> dict[str, Any]:
        """Search child chunks, rerank them, and load unique parent contexts."""
        logger.info("Retrieval started query_len=%s top_k=%s hybrid_k=%s", len(query), top_k, hybrid_k)
        vector_store = get_vector_store()
        dense_docs = vector_store.similarity_search(query, k=hybrid_k)
        bm25_docs = self.bm25_store.search(query, top_k=hybrid_k)
        child_docs = reciprocal_rank_fusion([dense_docs, bm25_docs], limit=30)
        if not child_docs:
            child_docs = dense_docs
        logger.info("Dense/BM25/RRF returned dense=%s bm25=%s fused=%s", len(dense_docs), len(bm25_docs), len(child_docs))
        reranked_docs, scores = rerank(query, child_docs, top_k=rerank_k)

        parent_docs: list[Document] = []
        seen_parent_ids: set[str] = set()
        for child in reranked_docs:
            parent_id = str(child.metadata.get("parent_id") or "")
            if not parent_id or parent_id in seen_parent_ids:
                continue
            parent = self.parent_store.load(parent_id)
            if parent:
                parent_docs.append(parent)
                seen_parent_ids.add(parent_id)
            if len(parent_docs) >= top_k:
                break

        citations = [_citation_from_metadata(doc.metadata) for doc in parent_docs]
        contexts = [doc.page_content for doc in parent_docs]
        trace = {
            "dense_child_count": len(dense_docs),
            "bm25_child_count": len(bm25_docs),
            "retrieved_child_count": len(child_docs),
            "reranked_count": len(reranked_docs),
            "parent_count": len(parent_docs),
            "rerank_scores": scores,
        }
        logger.info("Retrieval finished parents=%s citations=%s", len(parent_docs), len(citations))
        return {"contexts": contexts, "citations": citations, "trace": trace}
