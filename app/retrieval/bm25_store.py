"""Local PyVi-tokenized BM25 sidecar index for Vietnamese lexical retrieval."""

import logging
import pickle
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.retrieval.vietnamese_tokenizer import tokenize_vietnamese

logger = logging.getLogger(__name__)


class BM25Store:
    """Persist and query a local BM25 index built from child chunks."""

    # Khoi tao duong dan luu BM25 index local.
    def __init__(self, path: Path | None = None) -> None:
        """Create a BM25 store pointing at the configured pickle file."""
        self.path = path or get_settings().bm25_store_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._bm25: BM25Okapi | None = None
        self._documents: list[Document] = []

    # Build BM25 tu child chunks da duoc tao san trong artifact.
    def build(self, documents: list[Document]) -> None:
        """Build an in-memory BM25 index from child documents."""
        tokenized = [tokenize_vietnamese(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._documents = documents
        logger.info("Built BM25 index documents=%s", len(documents))

    # Luu BM25 index xuong file de lan sau khong can build lai.
    def save(self) -> None:
        """Persist the BM25 model and child documents to disk."""
        if self._bm25 is None:
            raise ValueError("BM25 index has not been built")
        with self.path.open("wb") as file:
            pickle.dump({"bm25": self._bm25, "documents": self._documents}, file)
        logger.info("Saved BM25 index path=%s documents=%s", self.path, len(self._documents))

    # Tai BM25 index da build tu file pickle.
    def load(self) -> bool:
        """Load a persisted BM25 index; return False when absent."""
        if not self.path.exists():
            logger.warning("BM25 index not found path=%s", self.path)
            return False
        with self.path.open("rb") as file:
            payload: dict[str, Any] = pickle.load(file)
        self._bm25 = payload["bm25"]
        self._documents = payload["documents"]
        logger.info("Loaded BM25 index path=%s documents=%s", self.path, len(self._documents))
        return True

    # Tim kiem lexical bang BM25 da tokenize tieng Viet.
    def search(self, query: str, top_k: int = 50) -> list[Document]:
        """Search BM25 and return top child documents with bm25_score metadata."""
        if self._bm25 is None and not self.load():
            return []
        assert self._bm25 is not None
        tokens = tokenize_vietnamese(query)
        scores = self._bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
        results: list[Document] = []
        for index in ranked_indices:
            if scores[index] <= 0:
                continue
            doc = self._documents[index]
            metadata = dict(doc.metadata)
            metadata["bm25_score"] = float(scores[index])
            results.append(Document(page_content=doc.page_content, metadata=metadata))
        logger.info("BM25 search query_len=%s results=%s", len(query), len(results))
        return results
