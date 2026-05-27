"""Simple callable retrieval tools used by agent code or future tool binding."""

from app.retrieval.parent_store import ParentStore
from app.retrieval.retriever import LegalRetriever


# Tim kiem tai lieu phap ly bang retriever.
def search_legal_documents(query: str, top_k: int = 5) -> dict:
    """Run the high-level legal retriever for a query."""
    return LegalRetriever().search(query, top_k=top_k)


# Lay full parent context theo parent_id.
def retrieve_parent_context(parent_id: str) -> str:
    """Load the full parent context text for one parent id."""
    doc = ParentStore().load(parent_id)
    return doc.page_content if doc else ""
