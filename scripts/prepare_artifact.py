"""Prepare a portable legal-tax artifact on Colab or any stronger machine."""

import argparse
import logging
from pathlib import Path

from app.core.logging import configure_logging
from app.ingestion.artifact_io import document_to_row, ensure_artifact_dir, write_jsonl, write_stats
from app.ingestion.chunker import chunk_sections
from app.ingestion.document_sampler import distribution, sample_target_metadata
from app.ingestion.hf_loader import load_metadata_rows, stream_selected_content
from app.ingestion.html_cleaner import clean_html_to_text
from app.ingestion.legal_parser import parse_legal_sections

logger = logging.getLogger(__name__)


# Parse tham so CLI cho viec tao artifact.
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for artifact preparation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-documents", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/legal_tax_v1_50"))
    return parser.parse_args()


# Chay pipeline preprocess: metadata -> content -> clean -> parse -> chunk -> JSONL.
def main() -> None:
    """Build selected metadata, parent chunks, child chunks, and stats files."""
    args = parse_args()
    configure_logging()
    output_dir = ensure_artifact_dir(args.output_dir)

    rows = load_metadata_rows()
    selected = sample_target_metadata(rows, max_documents=args.max_documents)
    write_jsonl(output_dir / "selected_metadata.jsonl", selected)

    parent_rows: list[dict] = []
    child_rows: list[dict] = []
    processed = 0
    errors: list[str] = []

    for loaded in stream_selected_content(selected):
        try:
            cleaned = clean_html_to_text(loaded.content_html, doc_id=loaded.doc_id)
            sections = parse_legal_sections(cleaned, loaded.metadata)
            parents, children = chunk_sections(sections)
            parent_rows.extend(document_to_row(doc) for doc in parents)
            child_rows.extend(document_to_row(doc) for doc in children)
            processed += 1
            if processed % 25 == 0:
                logger.info("Artifact progress documents=%s parents=%s children=%s", processed, len(parent_rows), len(child_rows))
        except Exception as exc:
            logger.exception("Artifact preparation failed doc_id=%s", loaded.doc_id)
            errors.append(f"{loaded.doc_id}: {exc}")

    write_jsonl(output_dir / "parents.jsonl", parent_rows)
    write_jsonl(output_dir / "children.jsonl", child_rows)
    write_stats(
        output_dir / "stats.json",
        {
            "selected_documents": len(selected),
            "processed_documents": processed,
            "parent_chunks": len(parent_rows),
            "child_chunks": len(child_rows),
            "errors": errors,
            "doc_type_distribution": distribution(selected, "loai_van_ban"),
            "authority_distribution": distribution(selected, "co_quan_ban_hanh"),
            "status_distribution": distribution(selected, "tinh_trang_hieu_luc"),
        },
    )
    logger.info("Artifact completed dir=%s processed=%s errors=%s", output_dir, processed, len(errors))


if __name__ == "__main__":
    main()
