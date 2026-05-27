"""Unit tests for parent-child chunk generation."""

from app.ingestion.chunker import chunk_sections
from app.ingestion.legal_parser import LegalSection


# Kiem tra chunker giu metadata article/citation trong child chunks.
def test_chunk_sections_keeps_parent_metadata():
    """Child chunks should preserve article and citation metadata."""
    section = LegalSection(
        parent_id="p1",
        doc_id="d1",
        chapter="Chương I",
        section=None,
        article_number="1",
        article_title="Phạm vi",
        text="Điều 1. Phạm vi\n1. Nội dung thứ nhất.\n2. Nội dung thứ hai.",
        metadata={"id": "d1", "title": "Doc", "parent_id": "p1", "tinh_trang_hieu_luc": "Còn hiệu lực"},
    )
    parents, children = chunk_sections([section])
    assert parents[0].metadata["parent_id"] == "p1"
    assert children
    assert children[0].metadata["article_number"] == "1"
