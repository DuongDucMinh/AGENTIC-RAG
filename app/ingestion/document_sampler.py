"""Score and sample legal metadata rows for small but useful artifacts."""

from typing import Any

from app.ingestion.hf_loader import is_target_metadata

TITLE_KEYWORDS = [
    "thuế",
    "phí",
    "lệ phí",
    "trước bạ",
    "hóa đơn",
    "giá trị gia tăng",
    "thu nhập doanh nghiệp",
    "thu nhập cá nhân",
]

DOC_TYPE_SCORE = {"Luật": 50, "Nghị định": 40, "Thông tư": 30, "Thông tư liên tịch": 20}
AUTHORITY_SCORE = {"Quốc hội": 40, "Chính phủ": 35, "Bộ Tài chính": 30, "Ủy ban thường vụ Quốc hội": 25, "Thủ tướng Chính phủ": 20}
STATUS_SCORE = {"Còn hiệu lực": 20, "Hết hiệu lực một phần": 10}


# Tinh diem metadata de uu tien van ban co gia tri demo cao.
def score_metadata(row: dict[str, Any]) -> int:
    """Score one metadata row so the sampler keeps high-value legal documents."""
    title = str(row.get("title") or "").lower()
    score = 0
    score += DOC_TYPE_SCORE.get(str(row.get("loai_van_ban") or "").strip(), 0)
    score += AUTHORITY_SCORE.get(str(row.get("co_quan_ban_hanh") or "").strip(), 0)
    score += STATUS_SCORE.get(str(row.get("tinh_trang_hieu_luc") or "").strip(), 0)
    score += sum(10 for keyword in TITLE_KEYWORDS if keyword in title)
    return score


# Loc domain roi sap xep theo diem de lay top N tai lieu tot nhat.
def sample_target_metadata(rows: list[dict[str, Any]], max_documents: int) -> list[dict[str, Any]]:
    """Filter target metadata and return the top N scored documents."""
    selected = [row for row in rows if is_target_metadata(row)]
    selected.sort(key=lambda row: (score_metadata(row), str(row.get("ngay_ban_hanh") or ""), str(row.get("id") or "")), reverse=True)
    return selected[:max_documents]


# Dem phan bo cac field chinh de ghi vao stats artifact.
def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    """Count values for one metadata field."""
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "").strip() or "UNKNOWN"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
