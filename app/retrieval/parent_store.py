"""File-based parent chunk store for easy debugging in v1."""

import json
import logging
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ParentStore:
    """Save and load parent chunks as one JSON file per parent id."""

    # Khoi tao thu muc luu parent chunks.
    def __init__(self, root: Path | None = None) -> None:
        """Create the store directory if it does not exist."""
        self.root = root or get_settings().parent_store_dir
        self.root.mkdir(parents=True, exist_ok=True)

    # Chuyen parent_id thanh duong dan JSON an toan.
    def _path(self, parent_id: str) -> Path:
        """Map a parent id to a safe JSON path."""
        safe_id = parent_id.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe_id}.json"

    # Luu nhieu parent Document thanh cac file JSON.
    def save_many(self, docs: list[Document]) -> int:
        """Persist multiple parent documents and return the saved count."""
        count = 0
        for doc in docs:
            parent_id = str(doc.metadata.get("parent_id") or "")
            if not parent_id:
                logger.warning("Skipping parent without parent_id")
                continue
            payload = {"page_content": doc.page_content, "metadata": doc.metadata}
            self._path(parent_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            count += 1
        logger.info("Saved parent documents count=%s dir=%s", count, self.root)
        return count

    # Doc lai parent Document theo parent_id.
    def load(self, parent_id: str) -> Document | None:
        """Load a parent document by id, returning None when absent."""
        path = self._path(parent_id)
        if not path.exists():
            logger.warning("Parent document not found parent_id=%s", parent_id)
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Document(page_content=data.get("page_content", ""), metadata=data.get("metadata", {}))

    # Xoa tat ca parent JSON trong store.
    def clear(self) -> None:
        """Delete all stored parent JSON files."""
        for path in self.root.glob("*.json"):
            path.unlink()
        logger.info("Cleared parent store dir=%s", self.root)
