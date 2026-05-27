"""Convert raw legal HTML into normalized plain text for parsing."""

import logging
import re
import unicodedata

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BLOCK_TAGS = ["p", "div", "br", "tr", "li", "table", "h1", "h2", "h3", "h4"]


# Clean HTML thanh plain text va giu lai marker phap ly quan trong.
def clean_html_to_text(html: str, doc_id: str = "") -> str:
    """Remove HTML noise while preserving legal markers such as Dieu/Muc/Chuong."""
    if not html:
        logger.warning("Empty HTML content doc_id=%s", doc_id)
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag_name in BLOCK_TAGS:
        for tag in soup.find_all(tag_name):
            tag.append("\n")

    text = soup.get_text(separator=" ")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"Điều\s+(\d+[a-zA-Z]?)\s*\.", r"Điều \1.", text)
    text = text.strip()

    if len(text) < 200:
        logger.warning("Cleaned text is very short doc_id=%s length=%s", doc_id, len(text))
    return text
