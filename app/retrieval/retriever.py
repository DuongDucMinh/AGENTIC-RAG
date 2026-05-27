"""High-level retrieval pipeline: hybrid search, rerank, parent context load."""

import logging
import time
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from app.core.config import get_settings
from app.retrieval.parent_store import ParentStore
from app.retrieval.qdrant_store import get_vector_store
from app.retrieval.reranker import rerank
from app.retrieval.bm25_store import BM25Store
from app.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)
KEYWORD_HEAVY_TERMS = (
    "muc thu",
    "mức thu",
    "ty le",
    "tỷ lệ",
    "%",
    "le phi truoc ba",
    "lệ phí trước bạ",
    "chứng từ",
    "chung tu",
    "hạch toán",
    "hach toan",
    "công khai",
    "cong khai",
)
NEGATIVE_LEGAL_TERMS = ("xử lý vi phạm", "khiếu nại", "thủ tục", "nộp đủ")
PARENT_NEGATIVE_TERMS = ("xử lý vi phạm", "khiếu nại", "thủ tục")
DOMAIN_TERMS = ("thuế", "thue", "phí", "phi", "lệ phí", "le phi", "trước bạ", "truoc ba")
NON_DOMAIN_HINTS = ("xây dựng", "xay dung", "nhà ở", "nha o", "bất động sản", "bat dong san")
INTENT_MUC_THU_TERMS = ("mức thu", "muc thu", "tỷ lệ", "ty le", "%")
INTENT_DOI_TUONG_TERMS = ("đối tượng", "doi tuong", "chịu", "chiu", "gồm", "gom")
INTENT_TRACH_NHIEM_TERMS = ("trách nhiệm", "trach nhiem", "tổ chức thu", "to chuc thu")
INTENT_NGUYEN_TAC_TERMS = ("nguyên tắc", "nguyen tac", "xác định", "xac dinh")
RESPONSIBILITY_DETAIL_TERMS = ("chứng từ", "chung tu", "hạch toán", "hach toan", "công khai", "cong khai", "quyết toán", "quyet toan")
VIOLATION_TERMS = ("xử phạt", "xu phat", "vi phạm", "vi pham", "xử lý vi phạm", "xu ly vi pham")


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


# Rut gon thong tin candidate de log ranking ma khong spam full text.
def _candidate_summary(doc: Document) -> dict[str, Any]:
    """Return a compact summary for retrieval debug logs."""
    metadata = doc.metadata
    return {
        "doc_id": metadata.get("doc_id"),
        "parent_id": metadata.get("parent_id"),
        "article_number": metadata.get("article_number"),
        "section": metadata.get("section"),
        "title": str(metadata.get("title", ""))[:80],
    }


# Phat hien query can uu tien lexical match manh hon semantic match.
def _is_keyword_heavy_query(query: str) -> bool:
    """Return True when the query contains explicit legal/rate keywords."""
    lowered = query.lower()
    return any(term in lowered for term in KEYWORD_HEAVY_TERMS)


# Log top candidate truoc/sau cac buoc fusion/rerank de debug ranking.
def _log_candidate_list(name: str, docs: list[Document], limit: int = 5) -> None:
    """Log a compact top-N summary of retrieval candidates."""
    logger.info("%s_top_%s=%s", name, min(limit, len(docs)), [_candidate_summary(doc) for doc in docs[:limit]])


# Dong goi top candidates vao trace de benchmark qua script/API.
def _trace_candidate_list(docs: list[Document], limit: int = 5) -> list[dict[str, Any]]:
    """Return compact candidate summaries for retrieval debug traces."""
    return [_candidate_summary(doc) for doc in docs[:limit]]


# Xac dinh query co nam trong domain thue/phi/le phi cua artifact hay khong.
def _is_domain_query(query: str) -> bool:
    """Return True when query appears to be inside tax/fee domain."""
    lowered = query.lower()
    if any(term in lowered for term in DOMAIN_TERMS):
        return True
    if any(term in lowered for term in NON_DOMAIN_HINTS):
        return False
    return False


# Gan intent retrieval de cham diem context dung muc tieu cau hoi.
def _detect_query_intent(query: str) -> str:
    """Classify query intent for lightweight retrieval scoring."""
    lowered = query.lower()
    if any(term in lowered for term in INTENT_TRACH_NHIEM_TERMS):
        return "trach_nhiem"
    if any(term in lowered for term in INTENT_DOI_TUONG_TERMS):
        return "doi_tuong"
    if any(term in lowered for term in INTENT_NGUYEN_TAC_TERMS):
        return "nguyen_tac"
    if any(term in lowered for term in INTENT_MUC_THU_TERMS):
        return "muc_thu"
    if any(term in lowered for term in INTENT_NGUYEN_TAC_TERMS):
        return "nguyen_tac"
    return "generic"


# Loai bo candidate trung lap de ranking va log de doc hon.
def _unique_by_chunk(docs: list[Document]) -> list[Document]:
    """Deduplicate candidates by chunk key while preserving original order."""
    unique_docs: list[Document] = []
    seen: set[str] = set()
    for doc in docs:
        chunk_key = str(doc.metadata.get("chunk_id") or doc.metadata.get("parent_id") or "")
        if not chunk_key or chunk_key in seen:
            continue
        seen.add(chunk_key)
        unique_docs.append(doc)
    return unique_docs


# Tinh bonus nhe de day dieu khoan khop y dinh len truoc ma khong can cross-encoder.
def _heuristic_keyword_score(query: str, doc: Document) -> float:
    """Compute a lightweight lexical-intent score for explicit legal queries."""
    lowered_query = query.lower()
    intent = _detect_query_intent(query)
    lowered_title = str(doc.metadata.get("article_title") or doc.metadata.get("section") or "").lower()
    lowered_prefix = doc.page_content[:400].lower()

    score = float(doc.metadata.get("rrf_score", 0.0))
    if intent == "muc_thu":
        if "mức thu" in lowered_title or "muc thu" in lowered_title:
            score += 10.0
        if "mức thu" in lowered_prefix or "muc thu" in lowered_prefix:
            score += 8.0
        if "%" in lowered_prefix or "tỷ lệ" in lowered_prefix or "ty le" in lowered_prefix:
            score += 6.0
    if "le phi truoc ba" in lowered_query or "lệ phí trước bạ" in lowered_query:
        if "lệ phí trước bạ" in lowered_title or "le phi truoc ba" in lowered_title:
            score += 4.0
        if "lệ phí trước bạ" in lowered_prefix or "le phi truoc ba" in lowered_prefix:
            score += 2.0
    if intent == "doi_tuong":
        if "đối tượng chịu" in lowered_title or "doi tuong chiu" in lowered_title:
            score += 12.0
        if "đối tượng chịu" in lowered_prefix or "doi tuong chiu" in lowered_prefix:
            score += 8.0
        if "trách nhiệm" in lowered_title or "trach nhiem" in lowered_title:
            score -= 4.0
    if intent == "trach_nhiem":
        if "trách nhiệm" in lowered_title or "trach nhiem" in lowered_title:
            score += 12.0
        if "tổ chức thu" in lowered_title or "to chuc thu" in lowered_title:
            score += 8.0
        if "đối tượng chịu" in lowered_title or "doi tuong chiu" in lowered_title:
            score -= 4.0
    if intent == "nguyen_tac":
        if "nguyên tắc" in lowered_title or "nguyen tac" in lowered_title:
            score += 12.0
        if "xác định mức thu" in lowered_title or "xac dinh muc thu" in lowered_title:
            score += 9.0
        if "trách nhiệm" in lowered_title or "trach nhiem" in lowered_title:
            score -= 4.0
    if lowered_prefix.startswith("điều") or lowered_prefix.startswith("dieu"):
        score += 2.0
    if any(term in lowered_query for term in RESPONSIBILITY_DETAIL_TERMS):
        if any(term in lowered_prefix or term in lowered_title for term in RESPONSIBILITY_DETAIL_TERMS):
            score += 14.0
        if "trách nhiệm" in lowered_title or "trach nhiem" in lowered_title:
            score += 10.0
        if "tổ chức thu" in lowered_prefix or "to chuc thu" in lowered_prefix:
            score += 6.0
    if not any(term in lowered_query for term in VIOLATION_TERMS):
        if any(term in lowered_title for term in VIOLATION_TERMS):
            score -= 10.0
        if any(term in lowered_prefix[:250] for term in VIOLATION_TERMS):
            score -= 6.0
    if "xử lý vi phạm" in lowered_title or "xu ly vi pham" in lowered_title:
        score -= 10.0
    if "khiếu nại" in lowered_title or "khieu nai" in lowered_title:
        score -= 8.0
    if any(term in lowered_prefix for term in NEGATIVE_LEGAL_TERMS):
        score -= 4.0
    return score


# Sap lai candidates bang heuristic lexical nhe cho query explicit.
def _heuristic_rank(query: str, docs: list[Document], limit: int) -> tuple[list[Document], list[float]]:
    """Apply a cheap intent-aware rerank without loading a cross-encoder."""
    ranked = sorted(
        ((doc, _heuristic_keyword_score(query, doc)) for doc in docs),
        key=lambda item: item[1],
        reverse=True,
    )
    top_ranked = ranked[:limit]
    return [doc for doc, _ in top_ranked], [score for _, score in top_ranked]


# Cham diem parent context de loc top parent gon hon cho answer generation.
def _parent_intent_score(query: str, parent_doc: Document, source_score: float) -> float:
    """Compute a lightweight parent-level score for final context selection."""
    lowered_query = query.lower()
    intent = _detect_query_intent(query)
    article_title = str(parent_doc.metadata.get("article_title") or parent_doc.metadata.get("section") or "").lower()
    prefix = parent_doc.page_content[:600].lower()

    score = source_score
    if intent == "muc_thu":
        if "mức thu" in article_title or "muc thu" in article_title:
            score += 12.0
        if "mức thu" in prefix or "muc thu" in prefix:
            score += 8.0
        if "%" in prefix or "tỷ lệ" in prefix or "ty le" in prefix:
            score += 6.0
        if "đối tượng chịu" in article_title or "doi tuong chiu" in article_title:
            score -= 5.0
        if "trách nhiệm" in article_title or "trach nhiem" in article_title:
            score -= 6.0
    if intent == "doi_tuong":
        if "đối tượng chịu" in article_title or "doi tuong chiu" in article_title:
            score += 12.0
        if "đối tượng" in prefix or "doi tuong" in prefix:
            score += 6.0
        if "mức thu" in article_title or "muc thu" in article_title:
            score -= 3.0
    if intent == "trach_nhiem":
        if "trách nhiệm" in article_title or "trach nhiem" in article_title:
            score += 12.0
        if "tổ chức thu" in article_title or "to chuc thu" in article_title:
            score += 6.0
        if "đối tượng chịu" in article_title or "doi tuong chiu" in article_title:
            score -= 4.0
    if intent == "nguyen_tac":
        if "nguyên tắc" in article_title or "nguyen tac" in article_title:
            score += 12.0
        if "xác định mức thu" in article_title or "xac dinh muc thu" in article_title:
            score += 9.0
        if "trách nhiệm" in article_title or "trach nhiem" in article_title:
            score -= 4.0
    if "le phi truoc ba" in lowered_query or "lệ phí trước bạ" in lowered_query:
        if "lệ phí trước bạ" in article_title or "le phi truoc ba" in article_title:
            score += 4.0
        if "lệ phí trước bạ" in prefix or "le phi truoc ba" in prefix:
            score += 2.0
    if any(term in article_title or term in prefix for term in PARENT_NEGATIVE_TERMS):
        score -= 6.0
    if "xử lý vi phạm" in article_title or "xu ly vi pham" in article_title:
        score -= 10.0
    if "khiếu nại" in article_title or "khieu nai" in article_title:
        score -= 8.0
    return score


# Chon parent da xep hang theo huong da dang nguon (doc_id) truoc khi lay top_k.
def _select_diverse_parents(
    ranked_parents: list[tuple[Document, float]],
    limit: int,
    max_per_doc: int = 1,
) -> tuple[list[Document], list[float]]:
    """Select top parents while limiting duplicate documents in citations."""
    selected_docs: list[Document] = []
    selected_scores: list[float] = []
    per_doc_count: dict[str, int] = {}

    for doc, score in ranked_parents:
        doc_id = str(doc.metadata.get("doc_id") or "")
        if per_doc_count.get(doc_id, 0) >= max_per_doc:
            continue
        selected_docs.append(doc)
        selected_scores.append(score)
        per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1
        if len(selected_docs) >= limit:
            return selected_docs, selected_scores

    for doc, score in ranked_parents:
        key = str(doc.metadata.get("parent_id") or "")
        if any(str(existing.metadata.get("parent_id") or "") == key for existing in selected_docs):
            continue
        selected_docs.append(doc)
        selected_scores.append(score)
        if len(selected_docs) >= limit:
            break
    return selected_docs, selected_scores


# Sap lai parent contexts de uu tien cac dieu khoan dung y dinh cau hoi.
def _rank_parent_contexts(query: str, parent_candidates: list[tuple[Document, float]], limit: int) -> tuple[list[Document], list[float]]:
    """Rank parent contexts for final answer generation and citation output."""
    ranked = sorted(parent_candidates, key=lambda item: item[1], reverse=True)
    return _select_diverse_parents(ranked, limit=limit, max_per_doc=2)


class LegalRetriever:
    """Retrieve legal contexts suitable for grounded answer generation."""

    # Khoi tao parent store va BM25 sidecar.
    def __init__(self) -> None:
        """Initialize access to the parent context store."""
        self.parent_store = ParentStore()
        self.bm25_store = BM25Store()

    # Chay dense search, BM25, RRF, rerank va load parent contexts.
    def search(self, query: str, top_k: int = 5, hybrid_k: int | None = None, rerank_k: int | None = None) -> dict[str, Any]:
        """Search child chunks, rerank them, and load unique parent contexts."""
        search_start = time.perf_counter()
        settings = get_settings()
        if not _is_domain_query(query):
            elapsed_ms = (time.perf_counter() - search_start) * 1000
            logger.info("Retrieval skipped out_of_domain query=%s", query[:120])
            logger.info("Retrieval finished out_of_domain=true latency_ms=%.2f", elapsed_ms)
            return {
                "contexts": [],
                "citations": [],
                "trace": {
                    "out_of_domain": True,
                    "reason": "query_not_in_tax_fee_domain",
                    "keyword_heavy_query": False,
                },
            }
        dense_k = hybrid_k or settings.retrieval_dense_top_k
        bm25_k = settings.retrieval_bm25_top_k
        fusion_k = settings.retrieval_fusion_top_k
        final_rerank_k = rerank_k or settings.retrieval_rerank_top_k
        keyword_heavy = _is_keyword_heavy_query(query)
        query_intent = _detect_query_intent(query)
        bm25_weight = max(1, settings.retrieval_bm25_weight + (1 if keyword_heavy else 0))
        corpus_child_count = self.bm25_store.document_count()

        logger.info(
            "Retrieval started query_len=%s top_k=%s dense_k=%s bm25_k=%s fusion_k=%s rerank_k=%s keyword_heavy=%s intent=%s corpus_child_count=%s",
            len(query),
            top_k,
            dense_k,
            bm25_k,
            fusion_k,
            final_rerank_k,
            keyword_heavy,
            query_intent,
            corpus_child_count,
        )
        vector_store = get_vector_store()
        dense_start = time.perf_counter()
        dense_docs = _unique_by_chunk(vector_store.similarity_search(query, k=dense_k))
        dense_elapsed_ms = (time.perf_counter() - dense_start) * 1000
        bm25_start = time.perf_counter()
        bm25_docs = _unique_by_chunk(self.bm25_store.search(query, top_k=bm25_k))
        bm25_elapsed_ms = (time.perf_counter() - bm25_start) * 1000
        _log_candidate_list("dense", dense_docs)
        _log_candidate_list("bm25", bm25_docs)
        logger.info(
            "Retrieval stage latency dense_ms=%.2f bm25_ms=%.2f dense_docs=%s bm25_docs=%s",
            dense_elapsed_ms,
            bm25_elapsed_ms,
            len(dense_docs),
            len(bm25_docs),
        )

        fusion_start = time.perf_counter()
        child_docs = reciprocal_rank_fusion(
            [dense_docs, bm25_docs],
            limit=fusion_k,
            k=settings.retrieval_rrf_k,
            weights=[1, bm25_weight],
        )
        if not child_docs:
            child_docs = dense_docs
        child_docs = _unique_by_chunk(child_docs)
        fusion_elapsed_ms = (time.perf_counter() - fusion_start) * 1000
        logger.info("Dense/BM25/RRF returned dense=%s bm25=%s fused=%s bm25_weight=%s", len(dense_docs), len(bm25_docs), len(child_docs), bm25_weight)
        _log_candidate_list("rrf", child_docs)
        logger.info("Retrieval stage latency fusion_ms=%.2f fused_docs=%s", fusion_elapsed_ms, len(child_docs))

        skip_rerank = (
            not settings.enable_reranking
            or corpus_child_count <= settings.retrieval_skip_rerank_below_docs
        )
        if skip_rerank:
            if keyword_heavy:
                reranked_docs, scores = _heuristic_rank(query, child_docs, limit=final_rerank_k)
                logger.info("Heuristic rerank applied candidates=%s output=%s", len(child_docs), len(reranked_docs))
            else:
                reranked_docs = child_docs[:final_rerank_k]
                scores = []
            logger.info(
                "Reranking bypassed enabled=%s corpus_child_count=%s threshold=%s",
                settings.enable_reranking,
                corpus_child_count,
                settings.retrieval_skip_rerank_below_docs,
            )
            _log_candidate_list("heuristic", reranked_docs)
        else:
            reranked_docs, scores = rerank(query, child_docs, top_k=final_rerank_k)
            if keyword_heavy and not scores:
                reranked_docs, scores = _heuristic_rank(query, child_docs, limit=final_rerank_k)
                logger.info("Heuristic rerank applied after GPU rerank skip candidates=%s output=%s", len(child_docs), len(reranked_docs))
            _log_candidate_list("rerank", reranked_docs)

        parent_start = time.perf_counter()
        parent_candidates: list[tuple[Document, float]] = []
        seen_parent_ids: set[str] = set()
        for index, child in enumerate(reranked_docs):
            parent_id = str(child.metadata.get("parent_id") or "")
            if not parent_id or parent_id in seen_parent_ids:
                continue
            parent = self.parent_store.load(parent_id)
            if parent:
                score = scores[index] if index < len(scores) else float(len(reranked_docs) - index)
                parent_candidates.append((parent, score))
                seen_parent_ids.add(parent_id)
            if len(parent_candidates) >= max(top_k, len(reranked_docs)):
                break

        parent_docs, parent_scores = _rank_parent_contexts(query, parent_candidates, limit=top_k)
        parent_elapsed_ms = (time.perf_counter() - parent_start) * 1000
        citations = [_citation_from_metadata(doc.metadata) for doc in parent_docs]
        contexts = [doc.page_content for doc in parent_docs]
        trace = {
            "dense_child_count": len(dense_docs),
            "bm25_child_count": len(bm25_docs),
            "retrieved_child_count": len(child_docs),
            "reranked_count": len(reranked_docs),
            "parent_count": len(parent_docs),
            "rerank_scores": scores,
            "keyword_heavy_query": keyword_heavy,
            "query_intent": query_intent,
            "bm25_weight": bm25_weight,
            "corpus_child_count": corpus_child_count,
            "rerank_bypassed": skip_rerank,
            "parent_scores": parent_scores,
            "dense_top_candidates": _trace_candidate_list(dense_docs),
            "bm25_top_candidates": _trace_candidate_list(bm25_docs),
            "rrf_top_candidates": _trace_candidate_list(child_docs),
            "rerank_top_candidates": _trace_candidate_list(reranked_docs),
            "parent_top_candidates": _trace_candidate_list(parent_docs),
        }
        total_elapsed_ms = (time.perf_counter() - search_start) * 1000
        logger.info(
            "Retrieval stage latency parent_ms=%.2f parent_candidates=%s parent_docs=%s",
            parent_elapsed_ms,
            len(parent_candidates),
            len(parent_docs),
        )
        logger.info(
            "Retrieval finished parents=%s citations=%s latency_ms=%.2f",
            len(parent_docs),
            len(citations),
            total_elapsed_ms,
        )
        return {"contexts": contexts, "citations": citations, "trace": trace}


@lru_cache
def get_legal_retriever() -> LegalRetriever:
    """Return a cached retriever so BM25 and parent stores are reused per process."""
    return LegalRetriever()
