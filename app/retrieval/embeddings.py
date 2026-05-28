"""Factories for dense and sparse embedding models."""

import logging
import threading
from functools import lru_cache

import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from transformers import AutoModel, AutoTokenizer

from app.core.config import get_settings

logger = logging.getLogger(__name__)

E5_PREFIX = "intfloat/multilingual-e5"
_dense_embeddings_lock = threading.RLock()
_sparse_embeddings_lock = threading.RLock()


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply attention-mask-aware mean pooling to token embeddings."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TransformerEmbeddings(Embeddings):
    """Embed text with a Hugging Face encoder without sentence-transformers."""

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cpu"
        self._encode_lock = threading.RLock()
        logger.info("Loading dense embedding model=%s via transformers backend", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _prefix_text(prefix: str, text: str) -> str:
        stripped = text.strip()
        return stripped if not prefix or stripped.startswith(prefix) else f"{prefix}{stripped}"

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        with self._encode_lock:
            return self._encode_locked(texts, prefix)

    def _encode_locked(self, texts: list[str], prefix: str) -> list[list[float]]:
        encoded_vectors: list[list[float]] = []
        prefixed = [self._prefix_text(prefix, text) for text in texts]
        with torch.inference_mode():
            for start in range(0, len(prefixed), self.batch_size):
                batch = prefixed[start : start + self.batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                outputs = self.model(**tokens)
                pooled = _mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                encoded_vectors.extend(normalized.cpu().tolist())
        return encoded_vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self.document_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.query_prefix)[0]


@lru_cache
def _load_dense_embeddings() -> Embeddings:
    """Load and cache the dense embedding model used by Qdrant."""
    settings = get_settings()
    model_name = settings.embedding_model
    if model_name.startswith(E5_PREFIX):
        logger.info("Applying E5 query/passage prefixes for model=%s", model_name)
        return TransformerEmbeddings(
            model_name=model_name,
            query_prefix="query: ",
            document_prefix="passage: ",
        )
    return TransformerEmbeddings(model_name=model_name)


@lru_cache
def _load_sparse_embeddings() -> FastEmbedSparse:
    """Load and cache the sparse BM25-style embedding model."""
    settings = get_settings()
    logger.info("Loading sparse embedding model=%s", settings.sparse_model)
    return FastEmbedSparse(model_name=settings.sparse_model)


def get_dense_embeddings() -> Embeddings:
    """Return the process-wide dense embedding model singleton."""
    with _dense_embeddings_lock:
        return _load_dense_embeddings()


def get_sparse_embeddings() -> FastEmbedSparse:
    """Return the process-wide sparse embedding model singleton."""
    with _sparse_embeddings_lock:
        return _load_sparse_embeddings()
