"""Indexing endpoints for previewing filters and building the vector index."""

import logging

from fastapi import APIRouter, Query

from app.core.config import get_settings
from app.schemas.indexing import IndexingPreviewResponse, IndexingRunResponse, IndexingStatusResponse
from app.services.indexing_service import IndexingService

router = APIRouter(prefix="/indexing", tags=["indexing"])
logger = logging.getLogger(__name__)


@router.post("/preview", response_model=IndexingPreviewResponse)
# Xem truoc metadata sau filter ma chua stream content hay index.
def preview_indexing(limit: int = Query(default=20, ge=1, le=100)):
    """Preview selected metadata rows without streaming or indexing content."""
    logger.info("Indexing preview requested limit=%s", limit)
    return IndexingService().preview(limit=limit)


@router.post("/run", response_model=IndexingRunResponse)
# Chay indexing truc tiep tu Hugging Face dataset.
def run_indexing(max_documents: int | None = Query(default=None, ge=1), reset_collection: bool = False):
    """Run ingestion, parsing, chunking, parent storage, and Qdrant indexing."""
    settings = get_settings()
    resolved_max = max_documents or settings.max_documents_to_index
    logger.info("Indexing run requested max_documents=%s reset_collection=%s", resolved_max, reset_collection)
    return IndexingService().run(max_documents=resolved_max, reset_collection=reset_collection)


@router.get("/status", response_model=IndexingStatusResponse)
# Lay thong tin lan indexing gan nhat trong bo nho.
def indexing_status():
    """Return the latest in-memory indexing run summary."""
    logger.info("Indexing status requested")
    return IndexingService().status()
