"""Read and write portable preprocessing artifacts for Colab/local workflows."""

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# Tao thu muc artifact neu chua ton tai.
def ensure_artifact_dir(path: Path) -> Path:
    """Create and return an artifact directory."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Ghi tung object JSON thanh mot dong de file lon van doc streaming duoc.
def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write dictionaries to a JSONL file and return the row count."""
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    logger.info("Wrote JSONL path=%s rows=%s", path, count)
    return count


# Doc JSONL theo tung dong de khong can load het file lon vao RAM.
def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield dictionaries from a JSONL file."""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


# Chuyen LangChain Document thanh dict de ghi ra artifact JSONL.
def document_to_row(doc: Document) -> dict[str, Any]:
    """Convert a LangChain document into an artifact row."""
    return {"page_content": doc.page_content, "metadata": doc.metadata}


# Chuyen artifact row ve LangChain Document de import vao Qdrant/local store.
def row_to_document(row: dict[str, Any]) -> Document:
    """Convert an artifact row into a LangChain document."""
    return Document(page_content=row.get("page_content", ""), metadata=row.get("metadata", {}))


# Ghi file stats.json de xem artifact co bao nhieu docs/parents/children.
def write_stats(path: Path, stats: dict[str, Any]) -> None:
    """Write artifact statistics as pretty JSON."""
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote artifact stats path=%s", path)


# Doc stats.json neu can hien thi thong tin artifact.
def read_stats(path: Path) -> dict[str, Any]:
    """Read artifact statistics from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))
