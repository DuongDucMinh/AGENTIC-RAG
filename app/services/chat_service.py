"""Service wrapper around the LangGraph chat agent."""

import logging
from functools import lru_cache

from app.agent.graph import build_agent_graph
from app.core.tracing import tracing_context

logger = logging.getLogger(__name__)


class ChatService:
    """Build and invoke the RAG agent for chat requests."""

    # Compile LangGraph agent cho service instance.
    def __init__(self) -> None:
        """Compile a LangGraph agent instance for this service object."""
        self.graph = build_agent_graph()

    # Chay mot chat turn va tra ve answer/citation/debug trace.
    def chat(self, session_id: str, question: str, debug: bool = False) -> dict:
        """Run one chat turn and optionally include retrieval trace."""
        logger.info("Chat requested session_id=%s question_len=%s debug=%s", session_id, len(question), debug)
        with tracing_context("chat", tags=["chat", "legal-tax-rag", "langgraph"], metadata={"session_id": session_id, "debug": debug}):
            result = self.graph.invoke(
                {"session_id": session_id, "question": question},
                config={"tags": ["chat", "legal-tax-rag"], "metadata": {"session_id": session_id}},
            )
        trace = result.get("retrieval_trace") if debug else None
        return {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "out_of_domain": bool(result.get("out_of_domain")),
            "retrieval_trace": trace,
        }


@lru_cache
def get_chat_service() -> ChatService:
    """Return a cached chat service so the graph is compiled once per process."""
    return ChatService()
