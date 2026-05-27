"""Create child chunks from legal parent sections for vector search."""

import hashlib
import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.ingestion.legal_parser import LegalSection

logger = logging.getLogger(__name__)

CLAUSE_OR_POINT_RE = re.compile(r"(?m)^(?=(?:\d+\.|[a-zđ]\)))")


# Tao chunk_id on dinh cho child chunk.
def _chunk_id(parent_id: str, index: int, text: str) -> str:
    """Create deterministic child chunk ids from parent id and content."""
    digest = hashlib.sha1(f"{parent_id}:{index}:{text[:80]}".encode("utf-8")).hexdigest()[:12]
    return f"{parent_id}_child_{digest}"


# Tao metadata citation/filter dung chung cho parent va child.
def _base_metadata(section: LegalSection) -> dict[str, Any]:
    """Build citation and filtering metadata shared by parent and child chunks."""
    source = section.metadata
    return {
        "parent_id": section.parent_id,
        "doc_id": section.doc_id,
        "title": source.get("title", ""),
        "so_ky_hieu": source.get("so_ky_hieu", ""),
        "loai_van_ban": source.get("loai_van_ban", ""),
        "co_quan_ban_hanh": source.get("co_quan_ban_hanh", ""),
        "ngay_ban_hanh": source.get("ngay_ban_hanh", ""),
        "ngay_co_hieu_luc": source.get("ngay_co_hieu_luc", ""),
        "ngay_het_hieu_luc": source.get("ngay_het_hieu_luc", ""),
        "tinh_trang_hieu_luc": source.get("tinh_trang_hieu_luc", ""),
        "linh_vuc": source.get("linh_vuc", ""),
        "nganh": source.get("nganh", ""),
        "chapter": section.chapter,
        "section": section.section,
        "article_number": section.article_number,
        "article_title": section.article_title,
    }


# Chuyen LegalSection thanh parent Document.
def make_parent_document(section: LegalSection) -> Document:
    """Convert a legal section into a LangChain parent document."""
    return Document(page_content=section.text, metadata=_base_metadata(section))


# Tach mot parent section thanh cac child chunk de search.
def chunk_section(section: LegalSection) -> list[Document]:
    """Split one parent section into searchable child documents."""
    settings = get_settings()
    base = _base_metadata(section)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    clause_parts = [p.strip() for p in CLAUSE_OR_POINT_RE.split(section.text) if p.strip()]
    seed_docs: list[Document] = []
    if len(clause_parts) > 1:
        for part in clause_parts:
            seed_docs.append(Document(page_content=part, metadata=base.copy()))
    else:
        seed_docs = [Document(page_content=section.text, metadata=base.copy())]

    chunks: list[Document] = []
    for seed in seed_docs:
        if len(seed.page_content) > settings.child_chunk_size:
            chunks.extend(splitter.split_documents([seed]))
        else:
            chunks.append(seed)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = _chunk_id(section.parent_id, index, chunk.page_content)
    return chunks


# Tao danh sach parent docs va child docs cho mot van ban da parse.
def chunk_sections(sections: list[LegalSection]) -> tuple[list[Document], list[Document]]:
    """Return parent documents and all child chunks for a parsed document."""
    parent_docs = [make_parent_document(section) for section in sections]
    child_docs: list[Document] = []
    for section in sections:
        child_docs.extend(chunk_section(section))
    if sections:
        logger.info(
            "Chunked document doc_id=%s parents=%s children=%s",
            sections[0].doc_id,
            len(parent_docs),
            len(child_docs),
        )
    return parent_docs, child_docs
