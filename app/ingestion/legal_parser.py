"""Parse Vietnamese legal text into parent sections by article."""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

ARTICLE_RE = re.compile(r"(?m)^Điều\s+(\d+[a-zA-Z]?)\.\s*(.*)$")
CHAPTER_RE = re.compile(r"(?m)^Chương\s+([IVXLCDM]+|\d+)\b.*$", re.IGNORECASE)
SECTION_RE = re.compile(r"(?m)^Mục\s+(\d+)\b.*$", re.IGNORECASE)


@dataclass
class LegalSection:
    """One parent-level legal section, usually a single article."""
    parent_id: str
    doc_id: str
    chapter: str | None
    section: str | None
    article_number: str | None
    article_title: str | None
    text: str
    metadata: dict[str, Any]


# Tao parent_id on dinh de index lai khong bi doi id.
def _stable_parent_id(doc_id: str, article_number: str | None, index: int) -> str:
    """Create deterministic parent ids for repeatable indexing."""
    raw = f"{doc_id}:{article_number or 'fallback'}:{index}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}_parent_{digest}"


# Tim heading Chuong/Muc gan nhat nam truoc vi tri hien tai.
def _current_match_before(pattern: re.Pattern[str], text: str, pos: int) -> str | None:
    """Find the nearest chapter/section heading before a text position."""
    matches = [m.group(0).strip() for m in pattern.finditer(text, 0, pos)]
    return matches[-1] if matches else None


# Tach van ban da clean thanh cac parent section theo Dieu.
def parse_legal_sections(text: str, metadata: dict[str, Any]) -> list[LegalSection]:
    """Split a cleaned legal document into article-level parent sections."""
    doc_id = str(metadata.get("id") or metadata.get("doc_id") or "")
    article_matches = list(ARTICLE_RE.finditer(text))
    if article_matches:
        sections: list[LegalSection] = []
        for index, match in enumerate(article_matches):
            start = match.start()
            end = article_matches[index + 1].start() if index + 1 < len(article_matches) else len(text)
            article_text = text[start:end].strip()
            article_number = match.group(1).strip()
            article_title = match.group(2).strip() or None
            parent_id = _stable_parent_id(doc_id, article_number, index)
            section_metadata = {**metadata, "parent_id": parent_id, "article_number": article_number, "article_title": article_title}
            sections.append(
                LegalSection(
                    parent_id=parent_id,
                    doc_id=doc_id,
                    chapter=_current_match_before(CHAPTER_RE, text, start),
                    section=_current_match_before(SECTION_RE, text, start),
                    article_number=article_number,
                    article_title=article_title,
                    text=article_text,
                    metadata=section_metadata,
                )
            )
        logger.info("Parsed legal articles doc_id=%s article_count=%s", doc_id, len(sections))
        return sections

    logger.warning("No legal articles detected; using fallback parent splitting doc_id=%s", doc_id)
    return _fallback_sections(text, metadata)


# Fallback split theo do dai khi khong tim thay marker Dieu.
def _fallback_sections(text: str, metadata: dict[str, Any]) -> list[LegalSection]:
    """Create coarse parent sections when no article markers are detected."""
    settings = get_settings()
    doc_id = str(metadata.get("id") or metadata.get("doc_id") or "")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections: list[LegalSection] = []
    current: list[str] = []
    current_len = 0
    index = 0

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > settings.fallback_parent_chars:
            parent_id = _stable_parent_id(doc_id, None, index)
            parent_text = "\n\n".join(current)
            sections.append(
                LegalSection(parent_id=parent_id, doc_id=doc_id, chapter=None, section=None, article_number=None, article_title=None, text=parent_text, metadata={**metadata, "parent_id": parent_id})
            )
            index += 1
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph)

    if current:
        parent_id = _stable_parent_id(doc_id, None, index)
        parent_text = "\n\n".join(current)
        sections.append(
            LegalSection(parent_id=parent_id, doc_id=doc_id, chapter=None, section=None, article_number=None, article_title=None, text=parent_text, metadata={**metadata, "parent_id": parent_id})
        )

    logger.info("Created fallback legal sections doc_id=%s section_count=%s", doc_id, len(sections))
    return sections
