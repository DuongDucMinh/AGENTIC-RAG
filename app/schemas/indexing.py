"""Pydantic response models for indexing endpoints."""

from typing import Any

from pydantic import BaseModel


class IndexingPreviewResponse(BaseModel):
    """Preview result for metadata filtering before indexing content."""
    total_metadata_rows: int
    selected_count: int
    sample: list[dict[str, Any]]


class IndexingRunResponse(BaseModel):
    """Summary of one indexing job."""
    status: str
    documents_processed: int
    parents_stored: int
    child_chunks_indexed: int
    errors: list[str]


class IndexingStatusResponse(BaseModel):
    """Current indexing service status and last run summary."""
    status: str
    last_run: dict[str, Any] | None = None
