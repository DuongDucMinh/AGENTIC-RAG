"""Import a prepared artifact into local parent store, Qdrant, and BM25."""

import argparse
from pathlib import Path

from app.core.logging import configure_logging
from app.ingestion.artifact_io import read_jsonl, read_stats, row_to_document
from app.retrieval.bm25_store import BM25Store
from app.retrieval.parent_store import ParentStore
from app.retrieval.qdrant_store import add_child_documents, ensure_collection


# Parse tham so CLI cho import artifact ve local.
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for artifact import."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


# Import artifact: luu parent JSON, index child vao Qdrant, build BM25.
def main() -> None:
    """Import parent/child artifact files into the local retrieval stores."""
    args = parse_args()
    configure_logging()
    artifact_dir = args.artifact_dir
    parents_path = artifact_dir / "parents.jsonl"
    children_path = artifact_dir / "children.jsonl"
    stats_path = artifact_dir / "stats.json"

    if stats_path.exists():
        print(read_stats(stats_path))

    parent_docs = [row_to_document(row) for row in read_jsonl(parents_path)]
    child_docs = [row_to_document(row) for row in read_jsonl(children_path)]

    parent_store = ParentStore()
    if args.reset:
        parent_store.clear()
    parent_count = parent_store.save_many(parent_docs)

    ensure_collection(reset=args.reset)
    child_count = add_child_documents(child_docs)

    bm25_store = BM25Store()
    bm25_store.build(child_docs)
    bm25_store.save()

    print(f"parents_imported={parent_count}")
    print(f"children_indexed={child_count}")
    print(f"bm25_documents={len(child_docs)}")


if __name__ == "__main__":
    main()
