"""Pydantic models for the chat API."""

from typing import Any

from pydantic import BaseModel

from app.schemas.retrieval import Citation


class ChatRequest(BaseModel):
    """User question and session id sent to the RAG agent."""
    session_id: str
    question: str
    debug: bool = False


class ChatResponse(BaseModel):
    """Final assistant answer, citations, and optional retrieval trace."""
    answer: str
    citations: list[Citation]
    out_of_domain: bool = False
    retrieval_trace: dict[str, Any] | None = None
