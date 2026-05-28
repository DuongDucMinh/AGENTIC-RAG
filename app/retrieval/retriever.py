"""High-level retrieval pipeline: hybrid search, rerank, parent context load."""

import logging
import time
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from app.core.config import get_settings
from app.retrieval.parent_store import ParentStore
from app.retrieval.ranking import (
    aggregate_child_scores,
    deduplicate_near_duplicate_chunks,
    diversity_stats,
    select_diverse_parents,
    select_pre_rerank_candidates,
    source_key,
    unique_by_chunk,
)
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
NON_DOMAIN_VIOLATION_HINTS = (
    "giao thông",
    "giao thong",
    "xe máy",
    "xe may",
    "ô tô",
    "o to",
    "oto",
    "bằng lái",
    "bang lai",
    "đăng kiểm",
    "dang kiem",
)
DOMAIN_ASSET_TERMS = (
    "xe máy",
    "xe may",
    "ô tô",
    "o to",
    "oto",
    "nhà",
    "nha",
    "đất",
    "dat",
    "súng săn",
    "sung san",
    "súng thể thao",
    "sung the thao",
    "tàu",
    "tau",
    "ca nô",
    "ca no",
    "du thuyền",
    "du thuyen",
    "tàu bay",
    "tau bay",
    "tài sản",
    "tai san",
)
DOMAIN_ACTION_TERMS = (
    "mức",
    "muc",
    "mức thu",
    "muc thu",
    "nộp",
    "nop",
    "thu",
    "chịu",
    "chiu",
    "đăng ký",
    "dang ky",
    "lần đầu",
    "lan dau",
    "tỷ lệ",
    "ty le",
    "%",
)
INTENT_MUC_THU_TERMS = ("mức thu", "muc thu", "tỷ lệ", "ty le", "%", "mức nào", "muc nao")
INTENT_RATE_QUESTION_TERMS = ("bao nhiêu", "bao nhieu", "mức nào", "muc nao", "theo mức", "theo muc", "tỷ lệ nào", "ty le nao")
INTENT_DOI_TUONG_TERMS = ("đối tượng", "doi tuong", "chịu", "chiu", "gồm", "gom")
INTENT_CAN_CU_TINH_TERMS = (
    "căn cứ",
    "can cu",
    "giá tính",
    "gia tinh",
    "giá trị đất",
    "gia tri dat",
    "trị giá nhà",
    "tri gia nha",
    "trị giá",
    "tri gia",
    "tính lệ phí",
    "tinh le phi",
)
INTENT_THAM_QUYEN_TERMS = ("thẩm quyền", "tham quyen", "ai quyết định", "ai quyet dinh", "quyết định mức", "quyet dinh muc")
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
    if any(term in lowered for term in VIOLATION_TERMS) and any(term in lowered for term in NON_DOMAIN_VIOLATION_HINTS):
        return False
    if any(term in lowered for term in NON_DOMAIN_HINTS):
        return False
    has_asset = any(term in lowered for term in DOMAIN_ASSET_TERMS)
    has_legal_action = any(term in lowered for term in DOMAIN_ACTION_TERMS)
    if has_asset and has_legal_action:
        return True
    return False


# Gan intent retrieval de cham diem context dung muc tieu cau hoi.
def _detect_query_intent(query: str) -> str:
    """Classify query intent for lightweight retrieval scoring."""
    lowered = query.lower()
    if any(term in lowered for term in INTENT_TRACH_NHIEM_TERMS):
        return "trach_nhiem"
    if any(term in lowered for term in INTENT_THAM_QUYEN_TERMS) or lowered.startswith("ai "):
        return "tham_quyen"
    if any(term in lowered for term in INTENT_CAN_CU_TINH_TERMS):
        return "can_cu_tinh"
    if any(term in lowered for term in INTENT_NGUYEN_TAC_TERMS):
        return "nguyen_tac"
    if any(term in lowered for term in INTENT_MUC_THU_TERMS):
        return "muc_thu"
    if any(term in lowered for term in INTENT_RATE_QUESTION_TERMS) and (
        any(term in lowered for term in DOMAIN_TERMS)
        or any(term in lowered for term in DOMAIN_ASSET_TERMS)
    ):
        return "muc_thu"
    if any(term in lowered for term in INTENT_DOI_TUONG_TERMS):
        return "doi_tuong"
    return "generic"


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
        if any(term in lowered_query for term in DOMAIN_ASSET_TERMS) and any(term in lowered_prefix for term in DOMAIN_ASSET_TERMS):
            score += 4.0
        if "đối tượng chịu" in lowered_title or "doi tuong chiu" in lowered_title:
            score -= 8.0
        if "đối tượng chịu" in lowered_prefix[:250] or "doi tuong chiu" in lowered_prefix[:250]:
            score -= 5.0
    if "le phi truoc ba" in lowered_query or "lệ phí trước bạ" in lowered_query:
        if "lệ phí trước bạ" in lowered_title or "le phi truoc ba" in lowered_title:
            score += 4.0
        if "lệ phí trước bạ" in lowered_prefix or "le phi truoc ba" in lowered_prefix:
            score += 2.0
    if intent == "doi_tuong":
        if "đối tượng chịu" in lowered_title or "doi tuong chiu" in lowered_title:
            score += 16.0
        if "đối tượng chịu" in lowered_prefix or "doi tuong chiu" in lowered_prefix:
            score += 8.0
        if "trách nhiệm" in lowered_title or "trach nhiem" in lowered_title:
            score -= 4.0
        if "người nộp" in lowered_title or "nguoi nop" in lowered_title:
            score -= 8.0
        if "phạm vi điều chỉnh" in lowered_title or "pham vi dieu chinh" in lowered_title:
            score -= 5.0
    if intent == "can_cu_tinh":
        if "căn cứ tính" in lowered_title or "can cu tinh" in lowered_title:
            score += 16.0
        if "căn cứ tính" in lowered_prefix or "can cu tinh" in lowered_prefix:
            score += 10.0
        if any(term in lowered_title or term in lowered_prefix for term in ("giá tính", "gia tinh", "giá trị", "gia tri", "trị giá", "tri gia")):
            score += 8.0
        if "nguyên tắc xác định mức thu" in lowered_title or "nguyen tac xac dinh muc thu" in lowered_title:
            score -= 8.0
        if "đối tượng chịu" in lowered_title or "doi tuong chiu" in lowered_title:
            score -= 5.0
    if intent == "tham_quyen":
        if "hội đồng nhân dân" in lowered_prefix or "hoi dong nhan dan" in lowered_prefix:
            score += 16.0
        if "ủy ban nhân dân" in lowered_prefix or "uy ban nhan dan" in lowered_prefix:
            score += 8.0
        if "mức thu" in lowered_title or "muc thu" in lowered_title:
            score += 4.0
        if "hội đồng xử lý" in lowered_title or "hoi dong xu ly" in lowered_title:
            score -= 10.0
        if "bồi thường" in lowered_title or "boi thuong" in lowered_title:
            score -= 8.0
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


def _candidate_pre_rerank_score(query: str, doc: Document, keyword_heavy: bool) -> float:
    """Score fused candidates for cheap diversity-aware preselection."""
    base_score = _heuristic_keyword_score(query, doc) if keyword_heavy else float(doc.metadata.get("rrf_score", 0.0))
    article_title = str(doc.metadata.get("article_title") or doc.metadata.get("section") or "").lower()
    article_number = str(doc.metadata.get("article_number") or "")
    if article_number:
        base_score += 0.15
    if article_title:
        base_score += 0.1
    return base_score


def _select_pre_rerank_candidates(query: str, docs: list[Document], limit: int, keyword_heavy: bool) -> list[Document]:
    """Select a more diverse candidate pool before reranking."""
    return select_pre_rerank_candidates(
        docs,
        limit=limit,
        score_fn=lambda doc: _candidate_pre_rerank_score(query, doc, keyword_heavy),
    )


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


def _build_parent_candidates(
    reranked_docs: list[Document],
    scores: list[float],
    parent_store: ParentStore,
) -> list[tuple[Document, float]]:
    """Aggregate reranked child evidence into parent-level candidates."""
    grouped_scores: dict[str, list[float]] = {}
    parent_children: dict[str, list[Document]] = {}

    for index, child in enumerate(reranked_docs):
        parent_id = str(child.metadata.get("parent_id") or "")
        if not parent_id:
            continue
        default_score = float(len(reranked_docs) - index)
        grouped_scores.setdefault(parent_id, []).append(scores[index] if index < len(scores) else default_score)
        parent_children.setdefault(parent_id, []).append(child)

    parent_candidates: list[tuple[Document, float]] = []
    for parent_id, child_scores in grouped_scores.items():
        parent = parent_store.load(parent_id)
        if not parent:
            continue
        metadata = dict(parent.metadata)
        metadata["support_child_count"] = len(child_scores)
        metadata["support_sources"] = sorted(
            {
                source_key(child)
                for child in parent_children.get(parent_id, [])
                if source_key(child)
            }
        )
        parent_candidates.append((Document(page_content=parent.page_content, metadata=metadata), aggregate_child_scores(child_scores)))
    return parent_candidates


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
        if any(term in lowered_query for term in DOMAIN_ASSET_TERMS) and any(term in prefix for term in DOMAIN_ASSET_TERMS):
            score += 4.0
        if "đối tượng chịu" in article_title or "doi tuong chiu" in article_title:
            score -= 8.0
        if "đối tượng chịu" in prefix[:250] or "doi tuong chiu" in prefix[:250]:
            score -= 5.0
        if "trách nhiệm" in article_title or "trach nhiem" in article_title:
            score -= 6.0
    if intent == "doi_tuong":
        if "đối tượng chịu" in article_title or "doi tuong chiu" in article_title:
            score += 16.0
        if "đối tượng" in prefix or "doi tuong" in prefix:
            score += 6.0
        if "mức thu" in article_title or "muc thu" in article_title:
            score -= 3.0
        if "người nộp" in article_title or "nguoi nop" in article_title:
            score -= 8.0
        if "phạm vi điều chỉnh" in article_title or "pham vi dieu chinh" in article_title:
            score -= 5.0
    if intent == "can_cu_tinh":
        if "căn cứ tính" in article_title or "can cu tinh" in article_title:
            score += 16.0
        if "căn cứ tính" in prefix or "can cu tinh" in prefix:
            score += 10.0
        if any(term in article_title or term in prefix for term in ("giá tính", "gia tinh", "giá trị", "gia tri", "trị giá", "tri gia")):
            score += 8.0
        if "nguyên tắc xác định mức thu" in article_title or "nguyen tac xac dinh muc thu" in article_title:
            score -= 8.0
        if "đối tượng chịu" in article_title or "doi tuong chiu" in article_title:
            score -= 5.0
    if intent == "tham_quyen":
        if "hội đồng nhân dân" in prefix or "hoi dong nhan dan" in prefix:
            score += 16.0
        if "ủy ban nhân dân" in prefix or "uy ban nhan dan" in prefix:
            score += 8.0
        if "mức thu" in article_title or "muc thu" in article_title:
            score += 4.0
        if "hội đồng xử lý" in article_title or "hoi dong xu ly" in article_title:
            score -= 10.0
        if "bồi thường" in article_title or "boi thuong" in article_title:
            score -= 8.0
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
    support_child_count = int(parent_doc.metadata.get("support_child_count") or 0)
    if support_child_count > 1:
        score += min(2.5, 0.7 * (support_child_count - 1))
    return score


# Sap lai parent contexts de uu tien cac dieu khoan dung y dinh cau hoi.
def _rank_parent_contexts(query: str, parent_candidates: list[tuple[Document, float]], limit: int) -> tuple[list[Document], list[float]]:
    """Rank parent contexts for final answer generation and citation output."""
    ranked = sorted(
        ((doc, _parent_intent_score(query, doc, score)) for doc, score in parent_candidates),
        key=lambda item: item[1],
        reverse=True,
    )
    return select_diverse_parents(ranked, limit=limit)


def _prefer_direct_rate_contexts(query: str, parent_docs: list[Document], parent_scores: list[float]) -> tuple[list[Document], list[float]]:
    """Keep only direct rate-table evidence for asset-specific rate questions."""
    lowered_query = query.lower()
    if _detect_query_intent(query) != "muc_thu":
        return parent_docs, parent_scores
    if not any(term in lowered_query for term in DOMAIN_ASSET_TERMS):
        return parent_docs, parent_scores
    if not any(term in lowered_query for term in INTENT_RATE_QUESTION_TERMS + INTENT_MUC_THU_TERMS):
        return parent_docs, parent_scores

    direct_docs: list[Document] = []
    direct_scores: list[float] = []
    for doc, score in zip(parent_docs, parent_scores):
        article_title = str(doc.metadata.get("article_title") or doc.metadata.get("section") or "").lower()
        prefix = doc.page_content[:700].lower()
        has_rate_table = (
            (
                "mức thu lệ phí trước bạ" in article_title
                or "muc thu le phi truoc ba" in article_title
                or prefix.startswith("điều 7. mức thu")
                or prefix.startswith("dieu 7. muc thu")
            )
            and ("%" in article_title or "%" in prefix or "tỷ lệ" in article_title or "ty le" in article_title)
        )
        if has_rate_table:
            direct_docs.append(doc)
            direct_scores.append(score)

    return (direct_docs, direct_scores) if direct_docs else (parent_docs, parent_scores)


def _prefer_direct_can_cu_contexts(query: str, parent_docs: list[Document], parent_scores: list[float]) -> tuple[list[Document], list[float]]:
    """Trim can-cu evidence to the legal slice asked by the query."""
    lowered_query = query.lower()
    if _detect_query_intent(query) != "can_cu_tinh":
        return parent_docs, parent_scores

    wants_components = any(term in lowered_query for term in ("thành phần", "thanh phan", "gồm", "gom"))
    wants_asset_value = any(term in lowered_query for term in ("giá trị đất", "gia tri dat", "trị giá nhà", "tri gia nha", "trị giá", "tri gia"))
    direct_docs: list[Document] = []
    direct_scores: list[float] = []
    for doc, score in zip(parent_docs, parent_scores):
        article_title = str(doc.metadata.get("article_title") or doc.metadata.get("section") or "").lower()
        prefix = doc.page_content[:700].lower()
        if wants_components and ("căn cứ tính lệ phí trước bạ" in article_title or "can cu tinh le phi truoc ba" in article_title):
            direct_docs.append(doc)
            direct_scores.append(score)
        elif wants_asset_value and any(term in article_title or term in prefix for term in ("căn cứ tính lệ phí trước bạ", "can cu tinh le phi truoc ba", "giá trị đất", "gia tri dat", "trị giá nhà", "tri gia nha")):
            direct_docs.append(doc)
            direct_scores.append(score)

    return (direct_docs, direct_scores) if direct_docs else (parent_docs, parent_scores)


def _prefer_direct_authority_contexts(query: str, parent_docs: list[Document], parent_scores: list[float]) -> tuple[list[Document], list[float]]:
    """Keep authority evidence when the question asks who decides a rate."""
    if _detect_query_intent(query) != "tham_quyen":
        return parent_docs, parent_scores

    direct_docs: list[Document] = []
    direct_scores: list[float] = []
    for doc, score in zip(parent_docs, parent_scores):
        text = (str(doc.metadata.get("article_title") or "") + " " + doc.page_content[:900]).lower()
        if "hội đồng nhân dân" in text or "hoi dong nhan dan" in text or "ủy ban nhân dân" in text or "uy ban nhan dan" in text:
            direct_docs.append(doc)
            direct_scores.append(score)

    return (direct_docs, direct_scores) if direct_docs else (parent_docs, parent_scores)


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
        keyword_heavy = _is_keyword_heavy_query(query)
        query_intent = _detect_query_intent(query)
        bm25_k = settings.retrieval_bm25_top_k
        fusion_k = settings.retrieval_fusion_top_k
        final_rerank_k = rerank_k or settings.retrieval_rerank_top_k
        if query_intent in {"muc_thu", "can_cu_tinh", "doi_tuong"}:
            bm25_k = max(bm25_k, 80)
            fusion_k = max(fusion_k, 60)
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
        from app.retrieval.qdrant_store import get_vector_store
        from app.retrieval.reranker import rerank

        vector_store_start = time.perf_counter()
        vector_store = get_vector_store()
        vector_store_elapsed_ms = (time.perf_counter() - vector_store_start) * 1000
        dense_start = time.perf_counter()
        dense_docs = unique_by_chunk(vector_store.similarity_search(query, k=dense_k))
        dense_elapsed_ms = (time.perf_counter() - dense_start) * 1000
        bm25_start = time.perf_counter()
        bm25_docs = unique_by_chunk(self.bm25_store.search(query, top_k=bm25_k))
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
        child_docs = unique_by_chunk(child_docs)
        deduped_child_docs = deduplicate_near_duplicate_chunks(child_docs)
        pre_rerank_limit = min(
            len(deduped_child_docs),
            max(final_rerank_k * 2, top_k * 3, min(fusion_k, 8)),
        )
        pre_rerank_docs = _select_pre_rerank_candidates(
            query,
            deduped_child_docs,
            limit=pre_rerank_limit,
            keyword_heavy=keyword_heavy,
        )
        fusion_elapsed_ms = (time.perf_counter() - fusion_start) * 1000
        logger.info(
            "Dense/BM25/RRF returned dense=%s bm25=%s fused=%s deduped=%s pre_rerank=%s bm25_weight=%s",
            len(dense_docs),
            len(bm25_docs),
            len(child_docs),
            len(deduped_child_docs),
            len(pre_rerank_docs),
            bm25_weight,
        )
        _log_candidate_list("rrf", child_docs)
        _log_candidate_list("pre_rerank", pre_rerank_docs)
        logger.info("Retrieval stage latency fusion_ms=%.2f fused_docs=%s", fusion_elapsed_ms, len(child_docs))

        rerank_start = time.perf_counter()
        skip_rerank = (
            not settings.enable_reranking
            or corpus_child_count <= settings.retrieval_skip_rerank_below_docs
        )
        if skip_rerank:
            if keyword_heavy:
                reranked_docs, scores = _heuristic_rank(query, pre_rerank_docs, limit=final_rerank_k)
                logger.info("Heuristic rerank applied candidates=%s output=%s", len(pre_rerank_docs), len(reranked_docs))
            else:
                reranked_docs = pre_rerank_docs[:final_rerank_k]
                scores = []
            logger.info(
                "Reranking bypassed enabled=%s corpus_child_count=%s threshold=%s",
                settings.enable_reranking,
                corpus_child_count,
                settings.retrieval_skip_rerank_below_docs,
            )
            _log_candidate_list("heuristic", reranked_docs)
        else:
            reranked_docs, scores = rerank(query, pre_rerank_docs, top_k=final_rerank_k)
            if keyword_heavy and not scores:
                reranked_docs, scores = _heuristic_rank(query, pre_rerank_docs, limit=final_rerank_k)
                logger.info("Heuristic rerank applied after GPU rerank skip candidates=%s output=%s", len(pre_rerank_docs), len(reranked_docs))
            _log_candidate_list("rerank", reranked_docs)
        rerank_elapsed_ms = (time.perf_counter() - rerank_start) * 1000

        parent_start = time.perf_counter()
        parent_candidates = _build_parent_candidates(reranked_docs, scores, self.parent_store)
        parent_docs, parent_scores = _rank_parent_contexts(query, parent_candidates, limit=top_k)
        parent_docs, parent_scores = _prefer_direct_rate_contexts(query, parent_docs, parent_scores)
        parent_docs, parent_scores = _prefer_direct_can_cu_contexts(query, parent_docs, parent_scores)
        parent_docs, parent_scores = _prefer_direct_authority_contexts(query, parent_docs, parent_scores)
        parent_elapsed_ms = (time.perf_counter() - parent_start) * 1000
        citations = [_citation_from_metadata(doc.metadata) for doc in parent_docs]
        contexts = [doc.page_content for doc in parent_docs]
        rrf_stats = diversity_stats(child_docs)
        pre_rerank_stats = diversity_stats(pre_rerank_docs)
        rerank_stats = diversity_stats(reranked_docs)
        parent_stats = diversity_stats(parent_docs)
        trace = {
            "dense_child_count": len(dense_docs),
            "bm25_child_count": len(bm25_docs),
            "retrieved_child_count": len(child_docs),
            "deduped_child_count": len(deduped_child_docs),
            "pre_rerank_count": len(pre_rerank_docs),
            "reranked_count": len(reranked_docs),
            "parent_count": len(parent_docs),
            "rerank_scores": scores,
            "keyword_heavy_query": keyword_heavy,
            "query_intent": query_intent,
            "bm25_weight": bm25_weight,
            "corpus_child_count": corpus_child_count,
            "rerank_bypassed": skip_rerank,
            "parent_scores": parent_scores,
            "vector_store_ms": vector_store_elapsed_ms,
            "dense_ms": dense_elapsed_ms,
            "bm25_ms": bm25_elapsed_ms,
            "fusion_ms": fusion_elapsed_ms,
            "rerank_ms": rerank_elapsed_ms,
            "parent_ms": parent_elapsed_ms,
            "dense_top_candidates": _trace_candidate_list(dense_docs),
            "bm25_top_candidates": _trace_candidate_list(bm25_docs),
            "rrf_top_candidates": _trace_candidate_list(child_docs),
            "pre_rerank_top_candidates": _trace_candidate_list(pre_rerank_docs),
            "rerank_top_candidates": _trace_candidate_list(reranked_docs),
            "parent_top_candidates": _trace_candidate_list(parent_docs),
            "rrf_diversity": rrf_stats,
            "pre_rerank_diversity": pre_rerank_stats,
            "rerank_diversity": rerank_stats,
            "parent_diversity": parent_stats,
        }
        total_elapsed_ms = (time.perf_counter() - search_start) * 1000
        trace["retrieval_total_ms"] = total_elapsed_ms
        logger.info(
            "Retrieval stage latency rerank_ms=%.2f parent_ms=%.2f parent_candidates=%s parent_docs=%s",
            rerank_elapsed_ms,
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
