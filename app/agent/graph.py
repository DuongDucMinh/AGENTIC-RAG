"""Build the LangGraph workflow used by the chat service."""

import logging

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import generate_answer_node, retrieve_node, rewrite_query_node, route_after_rewrite
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


# Build LangGraph workflow rewrite -> retrieve -> answer.
def build_agent_graph():
    """Compile the v1 graph: rewrite -> retrieve -> answer."""
    logger.info("Compiling LangGraph agent")
    graph = StateGraph(AgentState)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_edge(START, "rewrite_query")
    graph.add_conditional_edges("rewrite_query", route_after_rewrite, {"retrieve": "retrieve", "generate_answer": "generate_answer"})
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()
