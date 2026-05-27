"""Run HTML cleaning, legal parsing, and chunking on a small streamed sample."""

import argparse
from pathlib import Path

from app.core.logging import configure_logging
from app.ingestion.artifact_io import read_jsonl
from app.ingestion.chunker import chunk_sections
from app.ingestion.hf_loader import load_metadata_rows, select_target_metadata, stream_selected_content
from app.ingestion.html_cleaner import clean_html_to_text
from app.ingestion.legal_parser import parse_legal_sections


# In sample artifact local de tranh scan dataset goc tren may yeu.
def preview_artifact(artifact_dir: Path, max_documents: int) -> None:
    """Print parent/child samples from a prepared artifact directory."""
    parents = list(read_jsonl(artifact_dir / "parents.jsonl"))
    children = list(read_jsonl(artifact_dir / "children.jsonl"))
    print(f"artifact_dir={artifact_dir}")
    print(f"parents={len(parents)}")
    print(f"children={len(children)}")
    for index, row in enumerate(parents[:max_documents], start=1):
        metadata = row.get("metadata", {})
        print("=" * 80)
        print(f"sample_parent[{index}] parent_id={metadata.get('parent_id')}")
        print(f"title={metadata.get('title')}")
        print(f"article={metadata.get('article_number')} {metadata.get('article_title')}")
        print(f"chars={len(row.get('page_content', ''))}")
    if children:
        first = children[0]
        print("=" * 80)
        print(f"first_child_metadata={first.get('metadata', {})}")
        print(f"first_child_chars={len(first.get('page_content', ''))}")


# Chay parse truc tiep tu Hugging Face khi can debug pipeline goc.
def main() -> None:
    """Print parser/chunker output for a few selected legal documents."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-documents", type=int, default=3)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    if args.artifact_dir:
        preview_artifact(args.artifact_dir, args.max_documents)
        return

    rows = load_metadata_rows()
    selected = select_target_metadata(rows, max_documents=args.max_documents)
    for loaded in stream_selected_content(selected):
        cleaned = clean_html_to_text(loaded.content_html, doc_id=loaded.doc_id)
        sections = parse_legal_sections(cleaned, loaded.metadata)
        parents, children = chunk_sections(sections)
        print("=" * 80)
        print(f"doc_id={loaded.doc_id}")
        print(f"title={loaded.metadata.get('title')}")
        print(f"cleaned_chars={len(cleaned)}")
        print(f"parents={len(parents)}")
        print(f"children={len(children)}")
        if parents:
            print(f"first_parent_article={parents[0].metadata.get('article_number')}")
            print(f"first_parent_title={parents[0].metadata.get('article_title')}")
            print(f"first_parent_chars={len(parents[0].page_content)}")
        if children:
            print(f"first_child_chars={len(children[0].page_content)}")
            print(f"first_child_metadata={children[0].metadata}")


if __name__ == "__main__":
    main()
