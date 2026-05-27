"""Load and filter the Hugging Face Vietnamese legal dataset."""

import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from app.core.config import get_settings

logger = logging.getLogger(__name__)

TARGET_LINH_VUC = {
    "Quản lý thuế, phí, lệ phí và thu khác của ngân sách nhà nước",
    "Quản lý thuế, phí và lệ phí",
    "Chính sách thuế",
}
TARGET_NGANH = {"Tài chính"}
TARGET_AUTHORITIES = {
    "Bộ Tài chính",
    "Chính phủ",
    "Quốc hội",
    "Ủy ban thường vụ Quốc hội",
    "Thủ tướng Chính phủ",
}
TARGET_DOC_TYPES = {"Luật", "Nghị định", "Thông tư", "Thông tư liên tịch"}
TARGET_STATUS = {"Còn hiệu lực", "Hết hiệu lực một phần"}


@dataclass(frozen=True)
class LoadedLegalDocument:
    """Content row joined with its selected metadata row."""
    doc_id: str
    content_html: str
    metadata: dict[str, Any]


# Chuan hoa metadata value thanh string da strip.
def _clean_value(value: Any) -> str:
    """Normalize optional metadata values to stripped strings."""
    return str(value or "").strip()


# Kiem tra metadata co thuoc pham vi thue/phi/le phi v1 hay khong.
def is_target_metadata(row: dict[str, Any]) -> bool:
    """Return True when a metadata row belongs to the v1 tax/legal scope."""
    return (
        _clean_value(row.get("tinh_trang_hieu_luc")) in TARGET_STATUS
        and _clean_value(row.get("loai_van_ban")) in TARGET_DOC_TYPES
        and _clean_value(row.get("co_quan_ban_hanh")) in TARGET_AUTHORITIES
        and (
            _clean_value(row.get("linh_vuc")) in TARGET_LINH_VUC
            or _clean_value(row.get("nganh")) in TARGET_NGANH
        )
    )


# Load split metadata tu Hugging Face vao RAM.
def load_metadata_rows() -> list[dict[str, Any]]:
    """Load all metadata rows; this split is small enough for memory."""
    settings = get_settings()
    logger.info("Loading metadata dataset name=%s", settings.hf_dataset_name)
    dataset = load_dataset(settings.hf_dataset_name, "metadata", split="data")
    rows = [dict(row) for row in dataset]
    logger.info("Loaded metadata rows=%s", len(rows))
    return rows


# Dem top value cua mot field metadata de phan tich dataset.
def summarize_field(rows: list[dict[str, Any]], field: str, limit: int = 15) -> list[tuple[str, int]]:
    """Count the most common values for a metadata field."""
    counter = Counter(_clean_value(row.get(field)) for row in rows if _clean_value(row.get(field)))
    return counter.most_common(limit)


# Ap dung filter domain va gioi han so tai lieu neu can.
def select_target_metadata(rows: list[dict[str, Any]], max_documents: int | None = None) -> list[dict[str, Any]]:
    """Apply the v1 domain filter and optionally cap the number of documents."""
    selected = [row for row in rows if is_target_metadata(row)]
    selected.sort(key=lambda item: (_clean_value(item.get("ngay_ban_hanh")), _clean_value(item.get("id"))), reverse=True)
    if max_documents:
        selected = selected[:max_documents]
    logger.info("Selected target metadata rows=%s from total=%s", len(selected), len(rows))
    return selected


# Tra ve tong metadata, danh sach da chon va sample preview.
def preview_selection(limit: int = 20) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return total rows, selected rows, and a small sample for inspection."""
    rows = load_metadata_rows()
    selected = select_target_metadata(rows)
    return len(rows), selected, selected[:limit]


# Stream content_html va chi yield cac doc_id da duoc chon.
def stream_selected_content(selected_metadata: list[dict[str, Any]]) -> Iterator[LoadedLegalDocument]:
    """Stream content rows and yield only documents selected by metadata id."""
    settings = get_settings()
    selected_lookup = {_clean_value(row.get("id")): row for row in selected_metadata}
    logger.info("Streaming content dataset name=%s selected_ids=%s", settings.hf_dataset_name, len(selected_lookup))
    content_stream = load_dataset(settings.hf_dataset_name, "content", split="data", streaming=True)

    scanned = 0
    matched = 0
    for row in content_stream:
        scanned += 1
        doc_id = _clean_value(row.get("id"))
        if doc_id not in selected_lookup:
            if scanned % 10000 == 0:
                logger.info("Content stream progress scanned=%s matched=%s skipped=%s", scanned, matched, scanned - matched)
            continue
        matched += 1
        yield LoadedLegalDocument(
            doc_id=doc_id,
            content_html=str(row.get("content_html") or ""),
            metadata=selected_lookup[doc_id],
        )
        if matched == len(selected_lookup):
            break

    logger.info("Finished content stream scanned=%s matched=%s skipped=%s", scanned, matched, scanned - matched)
