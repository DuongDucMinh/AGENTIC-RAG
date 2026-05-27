"""Groq LLM factory and latency logging helpers."""

import logging
import time

from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.errors import ConfigurationError

logger = logging.getLogger(__name__)


# Tao Groq chat model va bao loi ro neu thieu API key.
def get_chat_model() -> ChatGroq:
    """Create a Groq chat model or raise a clear config error if key is missing."""
    settings = get_settings()
    if not settings.groq_api_key:
        raise ConfigurationError("GROQ_API_KEY is required for /chat. Set it in .env.")
    logger.info("Creating Groq chat model model=%s temperature=%s", settings.groq_model, settings.llm_temperature)
    return ChatGroq(model=settings.groq_model, temperature=settings.llm_temperature, api_key=settings.groq_api_key)


# Goi LLM va log latency.
def invoke_with_latency(messages):
    """Invoke the configured LLM and log request latency."""
    llm = get_chat_model()
    start = time.perf_counter()
    response = llm.invoke(messages)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("LLM call completed latency_ms=%.2f", elapsed_ms)
    return response
