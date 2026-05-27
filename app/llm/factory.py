"""Groq LLM factory and latency logging helpers."""

import logging
import time

from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.errors import ConfigurationError

logger = logging.getLogger(__name__)


# Tao Groq chat model va bao loi ro neu thieu API key.
def _require_api_key() -> str:
    """Return the configured Groq API key or raise a clear config error."""
    settings = get_settings()
    if not settings.groq_api_key:
        raise ConfigurationError("GROQ_API_KEY is required for /chat. Set it in .env.")
    return settings.groq_api_key


def _resolve_model_name(node_name: str) -> str:
    """Resolve a node-specific model name with backward-compatible fallback."""
    settings = get_settings()
    if node_name == "rewrite_query":
        return settings.groq_rewrite_model or settings.groq_model
    if node_name == "judge_context":
        return settings.groq_judge_model or settings.groq_model
    if node_name == "answer_with_citations":
        return settings.groq_answer_model or settings.groq_model
    return settings.groq_model


def _create_chat_model(node_name: str) -> ChatGroq:
    """Create a Groq chat model for a specific graph node."""
    settings = get_settings()
    model_name = _resolve_model_name(node_name)
    logger.info(
        "Creating Groq chat model node=%s model=%s temperature=%s",
        node_name,
        model_name,
        settings.llm_temperature,
    )
    return ChatGroq(
        model=model_name,
        temperature=settings.llm_temperature,
        api_key=_require_api_key(),
    )


def get_chat_model() -> ChatGroq:
    """Create a Groq chat model using the default answer model."""
    return _create_chat_model("answer_with_citations")


def get_rewrite_model() -> ChatGroq:
    """Create the Groq model used for query rewriting."""
    return _create_chat_model("rewrite_query")


def get_judge_model() -> ChatGroq:
    """Create the Groq model used for context sufficiency judgment."""
    return _create_chat_model("judge_context")


def get_answer_model() -> ChatGroq:
    """Create the Groq model used for final grounded answer generation."""
    return _create_chat_model("answer_with_citations")


# Goi LLM va log latency.
def invoke_with_latency(messages, node_name: str):
    """Invoke the configured node model and log request latency."""
    llm = _create_chat_model(node_name)
    start = time.perf_counter()
    response = llm.invoke(messages)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "LLM call completed node=%s model=%s latency_ms=%.2f",
        node_name,
        _resolve_model_name(node_name),
        elapsed_ms,
    )
    return response
