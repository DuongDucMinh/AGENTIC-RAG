"""Qdrant collection setup and child chunk indexing helpers."""

import logging

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.retrieval.embeddings import get_dense_embeddings, get_sparse_embeddings

logger = logging.getLogger(__name__)


# Tao Qdrant client tu URL trong config.
def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client from the configured URL."""
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


# Tao collection dense hoac dense+sparse tuy config.
def ensure_collection(reset: bool = False) -> None:
    """Create the hybrid dense+sparse collection, optionally resetting it."""
    settings = get_settings()
    client = get_qdrant_client()
    if reset and client.collection_exists(settings.qdrant_collection):
        logger.info("Deleting Qdrant collection name=%s", settings.qdrant_collection)
        client.delete_collection(settings.qdrant_collection)

    if client.collection_exists(settings.qdrant_collection):
        logger.info("Qdrant collection already exists name=%s", settings.qdrant_collection)
        return

    embeddings = get_dense_embeddings()
    dim = len(embeddings.embed_query("dimension probe"))
    logger.info("Creating Qdrant collection name=%s dense_dim=%s", settings.qdrant_collection, dim)
    create_kwargs = {
        "collection_name": settings.qdrant_collection,
        "vectors_config": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    }
    if settings.use_qdrant_sparse:
        create_kwargs["sparse_vectors_config"] = {"sparse": qmodels.SparseVectorParams()}
    client.create_collection(**create_kwargs)


# Tra ve LangChain QdrantVectorStore de add/search child chunks.
def get_vector_store() -> QdrantVectorStore:
    """Return a LangChain Qdrant vector store in hybrid retrieval mode."""
    settings = get_settings()
    ensure_collection(reset=False)
    if not settings.use_qdrant_sparse:
        return QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name=settings.qdrant_collection,
            embedding=get_dense_embeddings(),
            retrieval_mode=RetrievalMode.DENSE,
        )
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection,
        embedding=get_dense_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse",
    )


# Them child chunks vao Qdrant theo batch.
def add_child_documents(docs: list[Document], batch_size: int = 64) -> int:
    """Index child chunks into Qdrant in batches."""
    if not docs:
        return 0
    store = get_vector_store()
    total = 0
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        logger.info("Indexing Qdrant child batch start=%s size=%s", start, len(batch))
        store.add_documents(batch)
        total += len(batch)
    logger.info("Indexed Qdrant child chunks total=%s", total)
    return total
