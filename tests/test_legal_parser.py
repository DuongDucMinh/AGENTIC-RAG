"""Unit tests for legal article parsing."""

from app.ingestion.legal_parser import parse_legal_sections


# Kiem tra parser tach dung cac Dieu thanh parent sections.
def test_parse_legal_sections_detects_articles():
    """Parser should split text into article-level parent sections."""
    text = "Chương I\nQuy định chung\nĐiều 1. Phạm vi\nNội dung A\nĐiều 2. Đối tượng\nNội dung B"
    sections = parse_legal_sections(text, {"id": "doc1", "title": "Test"})
    assert len(sections) == 2
    assert sections[0].article_number == "1"
    assert sections[0].article_title == "Phạm vi"
