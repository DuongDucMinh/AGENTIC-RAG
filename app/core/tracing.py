"""Optional LangSmith tracing helpers for LangGraph and retrieval calls."""

import logging
import os
from contextlib import nullcontext
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_LANGSMITH_ENV_CONFIGURED = False


# Dong bo config LangSmith tu .env sang bien moi truong ma LangChain doc.
def configure_langsmith_environment() -> None:
    """Set LangSmith environment variables when tracing is enabled."""
    global _LANGSMITH_ENV_CONFIGURED
    settings = get_settings()
    if not settings.langsmith_tracing:
        return
    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is missing")
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    if not _LANGSMITH_ENV_CONFIGURED:
        logger.info("LangSmith tracing enabled project=%s endpoint=%s", settings.langsmith_project, settings.langsmith_endpoint or "default")
    _LANGSMITH_ENV_CONFIGURED = True


# Tao context tracing neu bat LangSmith, fallback nullcontext neu khong bat.
def tracing_context(name: str, tags: list[str] | None = None, metadata: dict[str, Any] | None = None):
    """Return a LangSmith tracing context when configured, otherwise a no-op."""
    settings = get_settings()
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return nullcontext()
    try:
        configure_langsmith_environment()
        import langsmith as ls

        return ls.tracing_context(project_name=settings.langsmith_project, enabled=True, tags=tags or [], metadata={"name": name, **(metadata or {})})
    except Exception as exc:
        logger.warning("Could not create LangSmith tracing context: %s", exc)
        return nullcontext()
