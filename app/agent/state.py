"""Shared state schema passed between LangGraph agent nodes."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Mutable state for one agent turn."""
    session_id: str
    question: str
    conversation_summary: str
    rewritten_query: str
    needs_clarification: bool
    clarification_question: str
    contexts: list[str]
    citations: list[dict[str, Any]]
    retrieval_trace: dict[str, Any]
    answer: str
