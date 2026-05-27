"""Reciprocal Rank Fusion for combining dense and BM25 rankings."""

from langchain_core.documents import Document


# Lay chunk_id lam key hop nhat, fallback sang parent_id/content neu thieu.
def document_key(doc: Document) -> str:
    """Return a stable key for fusing duplicate retrieval results."""
    return str(doc.metadata.get("chunk_id") or doc.metadata.get("parent_id") or doc.page_content[:120])


# Hop nhat nhieu bang xep hang bang RRF de can bang semantic va keyword search.
def reciprocal_rank_fusion(rankings: list[list[Document]], limit: int = 30, k: int = 60) -> list[Document]:
    """Fuse ranked document lists with RRF and return top unique documents."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = document_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, doc)
    ordered_keys = sorted(scores, key=scores.get, reverse=True)[:limit]
    fused: list[Document] = []
    for key in ordered_keys:
        doc = docs[key]
        metadata = dict(doc.metadata)
        metadata["rrf_score"] = scores[key]
        fused.append(Document(page_content=doc.page_content, metadata=metadata))
    return fused
