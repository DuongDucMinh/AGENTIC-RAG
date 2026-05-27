"""Vietnamese tokenization helpers for BM25 lexical retrieval."""

import re


# Tach tu tieng Viet bang PyVi, fallback sang regex neu PyVi chua cai duoc.
def tokenize_vietnamese(text: str) -> list[str]:
    """Tokenize Vietnamese text for BM25 indexing and search."""
    normalized = text.lower()
    try:
        from pyvi.ViTokenizer import tokenize

        normalized = tokenize(normalized)
    except Exception:
        normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return [token for token in normalized.split() if token.strip()]
