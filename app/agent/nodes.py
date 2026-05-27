"""LangGraph node functions for query rewriting, retrieval, and answering."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import ANSWER_PROMPT, REWRITE_PROMPT
from app.agent.state import AgentState
from app.llm.factory import invoke_with_latency
from app.retrieval.retriever import LegalRetriever

logger = logging.getLogger(__name__)


# Viet lai cau hoi hoac danh dau can clarification.
def rewrite_query_node(state: AgentState) -> AgentState:
    """Rewrite the user question or mark it as needing clarification."""
    logger.info("Agent node=rewrite_query session_id=%s", state.get("session_id"))
    prompt = f"Question:\n{state['question']}"
    response = invoke_with_latency([SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=prompt)])
    try:
        data = json.loads(str(response.content))
    except json.JSONDecodeError:
        logger.warning("Rewrite output was not JSON; using original query")
        data = {"needs_clarification": False, "rewritten_query": state["question"], "clarification_question": ""}
    state["needs_clarification"] = bool(data.get("needs_clarification"))
    state["rewritten_query"] = str(data.get("rewritten_query") or state["question"])
    state["clarification_question"] = str(data.get("clarification_question") or "Bạn muốn hỏi cụ thể về loại phí, lệ phí hoặc sắc thuế nào?")
    return state


# Truy xuat context phap ly cho rewritten query.
def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve parent legal contexts for the rewritten query."""
    logger.info("Agent node=retrieve session_id=%s", state.get("session_id"))
    result = LegalRetriever().search(state["rewritten_query"], top_k=5)
    state["contexts"] = result["contexts"]
    state["citations"] = result["citations"]
    trace = result["trace"]
    trace["rewritten_query"] = state["rewritten_query"]
    state["retrieval_trace"] = trace
    return state


# Sinh answer grounded hoac hoi lai neu cau hoi mo ho.
def generate_answer_node(state: AgentState) -> AgentState:
    """Generate a grounded answer, or ask for clarification when needed."""
    logger.info("Agent node=generate_answer session_id=%s context_count=%s", state.get("session_id"), len(state.get("contexts", [])))
    if state.get("needs_clarification"):
        state["answer"] = state.get("clarification_question", "Bạn có thể nói rõ hơn câu hỏi không?")
        state["citations"] = []
        state["retrieval_trace"] = {"path": "clarification"}
        return state

    context_text = "\n\n---\n\n".join(state.get("contexts", []))
    citation_text = json.dumps(state.get("citations", []), ensure_ascii=False)
    user_prompt = (
        f"QUESTION:\n{state['question']}\n\n"
        f"REWRITTEN_QUERY:\n{state.get('rewritten_query', '')}\n\n"
        f"CONTEXT:\n{context_text or 'NO_CONTEXT'}\n\n"
        f"CITATIONS_JSON:\n{citation_text}"
    )
    response = invoke_with_latency([SystemMessage(content=ANSWER_PROMPT), HumanMessage(content=user_prompt)])
    state["answer"] = str(response.content)
    return state


# Dieu huong sang retrieve neu cau hoi da ro, nguoc lai sang clarification.
def route_after_rewrite(state: AgentState) -> str:
    """Route to retrieval unless the rewritten question needs clarification."""
    if state.get("needs_clarification"):
        logger.info("Agent path=clarification session_id=%s", state.get("session_id"))
        return "generate_answer"
    logger.info("Agent path=retrieve session_id=%s", state.get("session_id"))
    return "retrieve"
