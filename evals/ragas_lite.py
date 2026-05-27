"""Custom lightweight keyword evaluation metrics without requiring Ragas runtime.

This is intentionally not a full RAGAS replacement. Use it as a smoke test for
retrieval/answer keyword regressions; use evals.run_eval as the main /chat gate.
"""

from typing import Any


# Kiem tra keyword mong doi co xuat hien trong retrieved contexts hay khong.
def retrieval_hit(contexts: list[str], expected_keywords: list[str]) -> bool:
    """Return True when all expected keywords appear in retrieved contexts."""
    if not expected_keywords:
        return True
    blob = "\n".join(contexts).lower()
    return all(keyword.lower() in blob for keyword in expected_keywords)


# Uoc luong context precision bang ty le context co keyword mong doi.
def context_precision_lite(contexts: list[str], expected_keywords: list[str]) -> float:
    """Compute a simple keyword-based context precision score."""
    if not contexts or not expected_keywords:
        return 0.0
    hits = 0
    for context in contexts:
        text = context.lower()
        if any(keyword.lower() in text for keyword in expected_keywords):
            hits += 1
    return hits / len(contexts)


# Kiem tra answer co citation hay khong.
def citation_present(citations: list[dict[str, Any]]) -> bool:
    """Return True when at least one citation is present."""
    return bool(citations)


# Kiem tra answer co bam keyword mong doi hay khong.
def answer_relevance_lite(answer: str, expected_keywords: list[str]) -> bool:
    """Return True when expected keywords appear in the answer."""
    if not expected_keywords:
        return True
    text = answer.lower()
    return any(keyword.lower() in text for keyword in expected_keywords)


# Kiem tra cau ngoai domain co tu choi/bao khong co du lieu thay vi bia.
def refusal_quality(answer: str) -> bool:
    """Return True when an out-of-domain answer clearly refuses or lacks data."""
    text = answer.lower()
    markers = ["không tìm thấy", "không có dữ liệu", "không đủ căn cứ", "ngoài phạm vi"]
    return any(marker in text for marker in markers)


# Gom cac metric lite thanh mot dict de ghi report.
def score_case(case: dict[str, Any], contexts: list[str], answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one evaluation case with lightweight RAG metrics."""
    expected = case.get("expected_keywords", [])
    case_type = case.get("type", "in_domain")
    scores = {
        "retrieval_hit": retrieval_hit(contexts, expected),
        "context_precision_lite": context_precision_lite(contexts, expected),
        "citation_present": citation_present(citations),
        "answer_relevance_lite": answer_relevance_lite(answer, expected),
        "refusal_quality": True,
        "mode": "custom_keyword_smoke_test",
    }
    if case_type == "out_of_domain":
        scores["refusal_quality"] = refusal_quality(answer)
    return scores
