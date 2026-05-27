"""Service that orchestrates dataset loading, parsing, chunking, and indexing."""

import logging
from typing import Any

from app.ingestion.chunker import chunk_sections
from app.ingestion.hf_loader import load_metadata_rows, preview_selection, select_target_metadata, stream_selected_content
from app.ingestion.html_cleaner import clean_html_to_text
from app.ingestion.legal_parser import parse_legal_sections
from app.retrieval.parent_store import ParentStore
from app.retrieval.qdrant_store import add_child_documents, ensure_collection

logger = logging.getLogger(__name__)

LAST_RUN: dict[str, Any] | None = None


class IndexingService:
    """Application service behind indexing API endpoints and CLI scripts."""

    # Xem truoc metadata duoc filter cho indexing.
    def preview(self, limit: int = 20) -> dict[str, Any]:
        """Preview selected metadata rows without streaming document content."""
        total, selected, sample = preview_selection(limit=limit)
        logger.info("Indexing preview total=%s selected=%s sample=%s", total, len(selected), len(sample))
        return {"total_metadata_rows": total, "selected_count": len(selected), "sample": sample}

    # Chay pipeline indexing truc tiep tu Hugging Face dataset.
    def run(self, max_documents: int, reset_collection: bool = False) -> dict[str, Any]:
        """Run a complete indexing job for a capped number of documents."""
        global LAST_RUN
        rows = load_metadata_rows()
        selected = select_target_metadata(rows, max_documents=max_documents)
        parent_store = ParentStore()
        if reset_collection:
            parent_store.clear()
        ensure_collection(reset=reset_collection)

        documents_processed = 0
        parents_stored = 0
        child_chunks_indexed = 0
        errors: list[str] = []

        for loaded_doc in stream_selected_content(selected):
            try:
                cleaned = clean_html_to_text(loaded_doc.content_html, doc_id=loaded_doc.doc_id)
                if not cleaned:
                    continue
                sections = parse_legal_sections(cleaned, loaded_doc.metadata)
                parent_docs, child_docs = chunk_sections(sections)
                parents_stored += parent_store.save_many(parent_docs)
                child_chunks_indexed += add_child_documents(child_docs)
                documents_processed += 1
                if documents_processed % 50 == 0:
                    logger.info("Indexing progress documents=%s parents=%s children=%s", documents_processed, parents_stored, child_chunks_indexed)
            except Exception as exc:
                logger.exception("Indexing failed doc_id=%s", loaded_doc.doc_id)
                errors.append(f"{loaded_doc.doc_id}: {exc}")

        LAST_RUN = {
            "documents_processed": documents_processed,
            "parents_stored": parents_stored,
            "child_chunks_indexed": child_chunks_indexed,
            "errors": errors,
        }
        logger.info("Indexing finished %s", LAST_RUN)
        return {"status": "completed", **LAST_RUN}

    # Tra ve summary cua lan indexing gan nhat.
    def status(self) -> dict[str, Any]:
        """Return the latest in-memory indexing summary."""
        return {"status": "ready", "last_run": LAST_RUN}
