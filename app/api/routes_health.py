"""Health endpoints for API and Qdrant readiness checks."""

import logging

from fastapi import APIRouter
from qdrant_client import QdrantClient

from app.core.config import get_settings

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
# Kiem tra API song va Qdrant co ket noi duoc hay khong.
def health_check():
    """Return API status and best-effort Qdrant connectivity status."""
    logger.info("Health check requested")
    settings = get_settings()
    qdrant = {"status": "unknown", "url": settings.qdrant_url}
    try:
        client = QdrantClient(url=settings.qdrant_url)
        client.get_collections()
        qdrant["status"] = "ok"
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        qdrant["status"] = "unreachable"
        qdrant["error"] = str(exc)

    return {"status": "ok", "qdrant": qdrant}
