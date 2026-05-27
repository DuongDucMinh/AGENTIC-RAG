"""LangGraph node functions for domain routing, rewrite, retrieval, judgment, and answering."""

import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import ANSWER_PROMPT, JUDGE_PROMPT, REWRITE_PROMPT
from app.agent.state import AgentState
from app.core.tracing import tracing_context
from app.llm.factory import invoke_with_latency
from app.retrieval.retriever import _is_domain_query, get_legal_retriever

logger = logging.getLogger(__name__)

BROAD_QUESTION_TERMS = (
    "được quy định như thế nào",
    "được quy định thế nào",
    "quy định như thế nào",
    "quy định thế nào",
    "quy định ra sao",
    "gồm những quy định gì",
)
VAGUE_REFERENCE_TERMS = ("mức phí đó", "lệ phí đó", "thuế đó", "đó là bao nhiêu", "cái này", "đó")
NARROW_SCOPE_TERMS = {
    "mức thu": "muc_thu",
    "tỷ lệ": "muc_thu",
    "căn cứ tính": "can_cu_tinh",
    "đối tượng": "doi_tuong",
    "trách nhiệm": "trach_nhiem",
    "nguyên tắc": "nguyen_tac",
    "thủ tục": "thu_tuc",
    "miễn": "mien_giam",
    "giảm": "mien_giam",
    "ưu đãi": "mien_giam",
}


def _node_metadata(state: AgentState, node_name: str) -> dict[str, Any]:
    """Build consistent tracing metadata for an agent node."""
    return {
        "session_id": state.get("session_id"),
        "node": node_name,
        "question_len": len(state.get("question", "")),
    }


def _trace_update(state: AgentState, **values: Any) -> None:
    """Merge structured debug fields into retrieval_trace."""
    trace = dict(state.get("retrieval_trace") or {})
    trace.update({key: value for key, value in values.items() if value is not None})
    state["retrieval_trace"] = trace


def _extract_text(value: Any) -> str:
    """Return text content from LangChain responses."""
    if isinstance(value, str):
        return value
    return str(getattr(value, "content", value))


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating code fences and surrounding text."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if "{" in text and "}" in text:
        text = text[text.find("{") : text.rfind("}") + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected object", text, 0)
    return parsed


def _is_broad_question(question: str) -> bool:
    """Return True when the user appears to ask for a broad legal overview."""
    lowered = question.lower()
    return any(term in lowered for term in BROAD_QUESTION_TERMS)


def _has_vague_reference(question: str) -> bool:
    """Return True when the question depends on a missing antecedent."""
    lowered = question.lower()
    return any(term in lowered for term in VAGUE_REFERENCE_TERMS)


def _infer_query_intent(question: str) -> str:
    """Infer a lightweight legal intent from the raw question text."""
    lowered = question.lower()
    if "trách nhiệm" in lowered or "trach nhiem" in lowered:
        return "trach_nhiem"
    if "đối tượng" in lowered or "doi tuong" in lowered:
        return "doi_tuong"
    if "nguyên tắc" in lowered or "nguyen tac" in lowered:
        return "nguyen_tac"
    if "mức thu" in lowered or "muc thu" in lowered or "tỷ lệ" in lowered or "ty le" in lowered or "%" in lowered:
        return "muc_thu"
    return "generic"


def _scope_labels_from_citations(citations: list[dict[str, Any]]) -> set[str]:
    """Infer coarse scope labels from citation titles and article titles."""
    labels: set[str] = set()
    for citation in citations:
        article_text = " ".join(
            str(citation.get(key) or "")
            for key in ("article_title", "title")
        ).lower()
        for term, label in NARROW_SCOPE_TERMS.items():
            if term in article_text:
                labels.add(label)
    return labels


def _contexts_cover_single_narrow_slice(question: str, citations: list[dict[str, Any]]) -> tuple[bool, str]:
    """Detect when a broad question retrieved only narrow topic slices."""
    if not _is_broad_question(question):
        return False, ""
    labels = _scope_labels_from_citations(citations)
    if len(labels) == 1:
        only_label = next(iter(labels))
        return True, f"context_only_covers_{only_label}"
    if 0 < len(labels) <= 2:
        return True, "context_scope_too_narrow_for_broad_question"
    if not labels and len(citations) <= 1:
        return True, "context_scope_too_narrow"
    return False, ""


def _judge_fallback(state: AgentState) -> dict[str, Any]:
    """Return a conservative fallback judgment when LLM output is unusable."""
    question = state.get("question", "")
    citations = state.get("citations", [])
    if not state.get("contexts") or not citations:
        return {
            "answerable": False,
            "insufficient_context": True,
            "needs_clarification": False,
            "clarification_question": "",
            "judge_reason": "no_usable_context",
        }
    narrow_slice, narrow_reason = _contexts_cover_single_narrow_slice(question, citations)
    if narrow_slice:
        return {
            "answerable": False,
            "insufficient_context": False,
            "needs_clarification": True,
            "clarification_question": "Bạn muốn hỏi cụ thể về đối tượng chịu, căn cứ tính, mức thu hay trách nhiệm liên quan đến nội dung này?",
            "judge_reason": narrow_reason,
        }
    return {
        "answerable": True,
        "insufficient_context": False,
        "needs_clarification": False,
        "clarification_question": "",
        "judge_reason": "fallback_answerable",
    }


def _has_direct_intent_support(state: AgentState) -> tuple[bool, str]:
    """Return True when citations directly support a specific legal intent."""
    question_intent = _infer_query_intent(state.get("question", ""))
    citations = state.get("citations", [])
    labels = _scope_labels_from_citations(citations)
    if question_intent == "trach_nhiem" and "trach_nhiem" in labels:
        return True, "direct_intent_match_trach_nhiem"
    if question_intent == "doi_tuong" and "doi_tuong" in labels:
        return True, "direct_intent_match_doi_tuong"
    if question_intent == "nguyen_tac" and "nguyen_tac" in labels:
        return True, "direct_intent_match_nguyen_tac"
    if question_intent == "muc_thu" and "muc_thu" in labels:
        return True, "direct_intent_match_muc_thu"
    return False, ""


def _deterministic_judgment(state: AgentState) -> dict[str, Any] | None:
    """Return a judgment without LLM when the case is obvious enough."""
    if not state.get("contexts") or not state.get("citations"):
        return {
            "answerable": False,
            "insufficient_context": True,
            "needs_clarification": False,
            "clarification_question": "",
            "judge_reason": "no_usable_context",
        }
    supported, reason = _has_direct_intent_support(state)
    if supported:
        return {
            "answerable": True,
            "insufficient_context": False,
            "needs_clarification": False,
            "clarification_question": "",
            "judge_reason": reason,
        }
    narrow_slice, narrow_reason = _contexts_cover_single_narrow_slice(
        state.get("question", ""),
        state.get("citations", []),
    )
    if narrow_slice:
        return {
            "answerable": False,
            "insufficient_context": False,
            "needs_clarification": True,
            "clarification_question": "Câu hỏi này khá rộng. Bạn muốn hỏi cụ thể về mức thu, căn cứ tính, đối tượng chịu hay trách nhiệm liên quan?",
            "judge_reason": narrow_reason,
        }
    return None


def _clarification_fallback(question: str) -> str:
    """Return a safe clarification prompt when rewrite/judge did not supply one."""
    if _has_vague_reference(question):
        return "Bạn muốn hỏi cụ thể về loại phí, lệ phí hoặc sắc thuế nào?"
    return "Bạn có thể nêu rõ hơn nội dung pháp lý cần hỏi, ví dụ mức thu, đối tượng chịu, căn cứ tính hay trách nhiệm liên quan?"


def _select_extractive_fallback_context(state: AgentState) -> str:
    """Pick the best available context snippet when synthesis LLM is unavailable."""
    contexts = state.get("contexts", [])
    citations = state.get("citations", [])
    if not contexts:
        return ""
    question_intent = _infer_query_intent(state.get("question", ""))
    preferred_labels = {
        "trach_nhiem": "trach_nhiem",
        "doi_tuong": "doi_tuong",
        "nguyen_tac": "nguyen_tac",
        "muc_thu": "muc_thu",
    }
    preferred = preferred_labels.get(question_intent)
    if preferred:
        for index, citation in enumerate(citations):
            article_text = " ".join(
                str(citation.get(key) or "")
                for key in ("article_title", "title")
            ).lower()
            for term, label in NARROW_SCOPE_TERMS.items():
                if term in article_text and label == preferred and index < len(contexts):
                    return contexts[index]
    return contexts[0]


def route_domain_node(state: AgentState) -> AgentState:
    """Deterministically route out-of-domain questions before any LLM call."""
    node_name = "route_domain"
    start = time.perf_counter()
    with tracing_context(node_name, tags=["chat", "langgraph", node_name], metadata=_node_metadata(state, node_name)):
        logger.info("Agent node=%s session_id=%s", node_name, state.get("session_id"))
        question = state.get("question", "")
        in_domain = _has_vague_reference(question) or _is_domain_query(question)
        state["out_of_domain"] = not in_domain
        state["out_of_domain_reason"] = "" if in_domain else "query_not_in_tax_fee_domain"
        state["response_mode"] = "out_of_domain" if not in_domain else ""
        _trace_update(
            state,
            out_of_domain=state["out_of_domain"],
            out_of_domain_reason=state["out_of_domain_reason"] or None,
            response_mode=state["response_mode"] or None,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Agent node=%s session_id=%s latency_ms=%.2f out_of_domain=%s", node_name, state.get("session_id"), elapsed_ms, state.get("out_of_domain"))
    return state


def rewrite_query_node(state: AgentState) -> AgentState:
    """Rewrite the user question lightly or mark it as needing clarification."""
    node_name = "rewrite_query"
    start = time.perf_counter()
    with tracing_context(node_name, tags=["chat", "langgraph", node_name], metadata=_node_metadata(state, node_name)):
        logger.info("Agent node=%s session_id=%s", node_name, state.get("session_id"))
        if not _has_vague_reference(state["question"]):
            data = {
                "needs_clarification": False,
                "rewritten_query": state["question"],
            }
        else:
            prompt = f"Question:\n{state['question']}"
            try:
                response = invoke_with_latency(
                    [SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=prompt)],
                    node_name=node_name,
                )
                data = _parse_json_object(_extract_text(response))
            except Exception:
                logger.warning("Rewrite step failed; using deterministic fallback", exc_info=True)
                data = {
                    "needs_clarification": _has_vague_reference(state["question"]),
                    "rewritten_query": state["question"],
                    "clarification_question": _clarification_fallback(state["question"]),
                }
        state["needs_clarification"] = bool(data.get("needs_clarification"))
        state["rewritten_query"] = str(data.get("rewritten_query") or state["question"])
        state["clarification_question"] = str(
            data.get("clarification_question") or _clarification_fallback(state["question"])
        )
        if state["needs_clarification"]:
            state["response_mode"] = "clarification"
        _trace_update(
            state,
            rewritten_query=state["rewritten_query"],
            needs_clarification=state["needs_clarification"],
            clarification_question=state["clarification_question"] if state["needs_clarification"] else None,
            response_mode=state.get("response_mode") or None,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Agent node=%s session_id=%s latency_ms=%.2f needs_clarification=%s",
        node_name,
        state.get("session_id"),
        elapsed_ms,
        state.get("needs_clarification"),
    )
    return state


def retrieve_context_node(state: AgentState) -> AgentState:
    """Retrieve legal contexts for the rewritten query."""
    node_name = "retrieve_context"
    start = time.perf_counter()
    with tracing_context(node_name, tags=["chat", "langgraph", node_name], metadata=_node_metadata(state, node_name)):
        logger.info("Agent node=%s session_id=%s", node_name, state.get("session_id"))
        query = state.get("rewritten_query") or state["question"]
        try:
            result = get_legal_retriever().search(query, top_k=3)
        except Exception as exc:
            logger.exception("Retrieval failed inside chat graph session_id=%s", state.get("session_id"))
            state["contexts"] = []
            state["citations"] = []
            state["answerable"] = False
            state["insufficient_context"] = True
            state["judge_reason"] = f"retrieval_error:{type(exc).__name__}"
            state["response_mode"] = "insufficient_context"
            _trace_update(
                state,
                rewritten_query=query,
                retrieval_error=f"{type(exc).__name__}: {exc}",
                judge_reason=state["judge_reason"],
                response_mode=state["response_mode"],
                insufficient_context=True,
            )
            return state
        state["contexts"] = result["contexts"]
        state["citations"] = result["citations"]
        trace = dict(result["trace"])
        trace["rewritten_query"] = query
        state["retrieval_trace"] = trace
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Agent node=%s session_id=%s latency_ms=%.2f context_count=%s citation_count=%s",
        node_name,
        state.get("session_id"),
        elapsed_ms,
        len(state.get("contexts", [])),
        len(state.get("citations", [])),
    )
    return state


def judge_context_node(state: AgentState) -> AgentState:
    """Judge whether the retrieved evidence is sufficient to answer the question."""
    node_name = "judge_context"
    start = time.perf_counter()
    with tracing_context(node_name, tags=["chat", "langgraph", node_name], metadata=_node_metadata(state, node_name)):
        logger.info("Agent node=%s session_id=%s context_count=%s", node_name, state.get("session_id"), len(state.get("contexts", [])))
        judgment = _deterministic_judgment(state)
        if judgment is None:
            context_summary = "\n\n---\n\n".join(context[:900] for context in state.get("contexts", [])[:3])
            citation_text = json.dumps(state.get("citations", [])[:5], ensure_ascii=False)
            user_prompt = (
                f"QUESTION:\n{state['question']}\n\n"
                f"REWRITTEN_QUERY:\n{state.get('rewritten_query', state['question'])}\n\n"
                f"CONTEXT_SUMMARY:\n{context_summary}\n\n"
                f"CITATIONS_JSON:\n{citation_text}"
            )
            try:
                response = invoke_with_latency(
                    [SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=user_prompt)],
                    node_name=node_name,
                )
                judgment = _parse_json_object(_extract_text(response))
            except Exception:
                logger.warning("Judge step failed; using conservative fallback", exc_info=True)
                judgment = _judge_fallback(state)

        state["answerable"] = bool(judgment.get("answerable"))
        state["insufficient_context"] = bool(judgment.get("insufficient_context"))
        state["needs_clarification"] = bool(judgment.get("needs_clarification"))
        if state["needs_clarification"]:
            state["clarification_question"] = str(
                judgment.get("clarification_question") or _clarification_fallback(state["question"])
            )
        state["judge_reason"] = str(judgment.get("judge_reason") or "judge_no_reason")

        if state["needs_clarification"]:
            state["answerable"] = False
            state["insufficient_context"] = False
            state["response_mode"] = "clarification"
        elif state["insufficient_context"] or not state["answerable"]:
            state["answerable"] = False
            state["insufficient_context"] = True
            state["response_mode"] = "insufficient_context"
        else:
            state["response_mode"] = "grounded_answer"

        _trace_update(
            state,
            judge_reason=state["judge_reason"],
            response_mode=state["response_mode"],
            answerable=state["answerable"],
            insufficient_context=state["insufficient_context"],
            needs_clarification=state["needs_clarification"],
            clarification_question=state.get("clarification_question") if state.get("needs_clarification") else None,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Agent node=%s session_id=%s latency_ms=%.2f response_mode=%s judge_reason=%s",
        node_name,
        state.get("session_id"),
        elapsed_ms,
        state.get("response_mode"),
        state.get("judge_reason"),
    )
    return state


def answer_with_citations_node(state: AgentState) -> AgentState:
    """Return the final grounded answer, clarification, refusal, or insufficient-context message."""
    node_name = "answer_with_citations"
    start = time.perf_counter()
    with tracing_context(node_name, tags=["chat", "langgraph", node_name], metadata=_node_metadata(state, node_name)):
        logger.info(
            "Agent node=%s session_id=%s mode=%s context_count=%s",
            node_name,
            state.get("session_id"),
            state.get("response_mode"),
            len(state.get("contexts", [])),
        )
        mode = state.get("response_mode") or ("out_of_domain" if state.get("out_of_domain") else "grounded_answer")

        if mode == "out_of_domain":
            state["answer"] = "Tôi hiện chỉ hỗ trợ câu hỏi về thuế, phí, lệ phí và các nghĩa vụ tài chính liên quan trong phạm vi dữ liệu đã index."
            state["citations"] = []
        elif mode == "clarification":
            state["answer"] = state.get("clarification_question") or _clarification_fallback(state.get("question", ""))
            state["citations"] = []
        elif mode == "insufficient_context":
            state["answer"] = "Tôi chưa có đủ căn cứ phù hợp trong dữ liệu đã index để trả lời chắc chắn câu hỏi này. Bạn có thể hỏi cụ thể hơn về mức thu, đối tượng chịu, căn cứ tính hoặc trách nhiệm liên quan."
            state["citations"] = []
        else:
            context_text = "\n\n---\n\n".join(state.get("contexts", []))
            citation_text = json.dumps(state.get("citations", []), ensure_ascii=False)
            user_prompt = (
                f"QUESTION:\n{state['question']}\n\n"
                f"REWRITTEN_QUERY:\n{state.get('rewritten_query', '')}\n\n"
                f"CONTEXT:\n{context_text or 'NO_CONTEXT'}\n\n"
                f"CITATIONS_JSON:\n{citation_text}"
            )
            try:
                response = invoke_with_latency(
                    [SystemMessage(content=ANSWER_PROMPT), HumanMessage(content=user_prompt)],
                    node_name=node_name,
                )
                state["answer"] = _extract_text(response)
            except Exception:
                logger.warning("Answer synthesis failed; using extractive fallback", exc_info=True)
                fallback_context = _select_extractive_fallback_context(state)
                state["answer"] = fallback_context[:900] if fallback_context else "Tôi chưa thể tổng hợp câu trả lời do dịch vụ mô hình đang tạm thời lỗi."

        _trace_update(
            state,
            response_mode=mode,
            out_of_domain=state.get("out_of_domain"),
            out_of_domain_reason=state.get("out_of_domain_reason") or None,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Agent node=%s session_id=%s latency_ms=%.2f mode=%s", node_name, state.get("session_id"), elapsed_ms, mode)
    return state


def route_after_domain(state: AgentState) -> str:
    """Route out-of-domain requests directly to the terminal answer node."""
    if state.get("out_of_domain"):
        logger.info("Agent path=out_of_domain session_id=%s", state.get("session_id"))
        return "answer_with_citations"
    logger.info("Agent path=rewrite session_id=%s", state.get("session_id"))
    return "rewrite_query"


def route_after_rewrite(state: AgentState) -> str:
    """Route to retrieval unless the question still needs clarification."""
    if state.get("needs_clarification"):
        logger.info("Agent path=clarification session_id=%s", state.get("session_id"))
        return "answer_with_citations"
    logger.info("Agent path=retrieve_context session_id=%s", state.get("session_id"))
    return "retrieve_context"


def route_after_judge(state: AgentState) -> str:
    """Route all post-judgment outcomes into the single exit node."""
    mode = state.get("response_mode") or "insufficient_context"
    logger.info("Agent path=%s session_id=%s", mode, state.get("session_id"))
    return "answer_with_citations"
