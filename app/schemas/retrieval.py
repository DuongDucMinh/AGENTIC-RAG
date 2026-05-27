"""Pydantic models for retrieval requests, citations, and responses."""

from typing import Any

from pydantic import BaseModel


class RetrievalSearchRequest(BaseModel):
    """Input payload for direct retrieval testing."""
    query: str
    top_k: int = 3
    debug: bool = True


class Citation(BaseModel):
    """Source metadata shown to users for grounded legal answers."""
    title: str = ""
    so_ky_hieu: str = ""
    article_number: str | None = None
    article_title: str | None = None
    status: str = ""
    doc_id: str = ""
    parent_id: str = ""


class RetrievalSearchResponse(BaseModel):
    """Retrieved parent contexts plus citations and optional debug trace."""
    query: str
    contexts: list[str]
    citations: list[Citation]
    trace: dict[str, Any]
