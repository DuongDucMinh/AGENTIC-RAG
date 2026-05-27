"""Build the LangGraph workflow used by the chat service."""

import logging

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    answer_with_citations_node,
    judge_context_node,
    retrieve_context_node,
    rewrite_query_node,
    route_after_domain,
    route_after_judge,
    route_domain_node,
    route_after_rewrite,
)
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


# Build LangGraph workflow route_domain -> rewrite -> retrieve -> judge -> answer.
def build_agent_graph():
    """Compile the production-lite graph with explicit routing and judgment."""
    logger.info("Compiling LangGraph agent")
    graph = StateGraph(AgentState)
    graph.add_node("route_domain", route_domain_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("judge_context", judge_context_node)
    graph.add_node("answer_with_citations", answer_with_citations_node)
    graph.add_edge(START, "route_domain")
    graph.add_conditional_edges(
        "route_domain",
        route_after_domain,
        {
            "rewrite_query": "rewrite_query",
            "answer_with_citations": "answer_with_citations",
        },
    )
    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "retrieve_context": "retrieve_context",
            "answer_with_citations": "answer_with_citations",
        },
    )
    graph.add_edge("retrieve_context", "judge_context")
    graph.add_conditional_edges(
        "judge_context",
        route_after_judge,
        {"answer_with_citations": "answer_with_citations"},
    )
    graph.add_edge("answer_with_citations", END)
    return graph.compile()
