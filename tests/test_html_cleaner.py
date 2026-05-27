"""Unit tests for HTML cleaning."""

from app.ingestion.html_cleaner import clean_html_to_text


# Kiem tra cleaner xoa tag/script nhung giu text phap ly.
def test_clean_html_to_text_removes_tags_and_keeps_content():
    """Cleaner should remove tags/scripts but keep legal text."""
    text = clean_html_to_text("<html><body><p>Điều 1. Nội dung</p><script>x</script></body></html>", doc_id="x")
    assert "<p>" not in text
    assert "Điều 1. Nội dung" in text
    assert "script" not in text
