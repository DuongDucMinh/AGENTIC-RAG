"""Pure ranking helpers for retrieval diversity, deduplication, and aggregation."""

import re
from typing import Callable

from langchain_core.documents import Document


def normalize_text(value: str) -> str:
    """Normalize text for cheap duplicate detection across overlapping chunks."""
    return re.sub(r"\s+", " ", value.strip().lower())


def source_key(doc: Document) -> str:
    """Return a stable source identity for diversity scoring."""
    metadata = doc.metadata
    return str(metadata.get("so_ky_hieu") or metadata.get("doc_id") or metadata.get("parent_id") or "")


def duplicate_fingerprint(doc: Document) -> str:
    """Return a coarse fingerprint to drop near-duplicate overlapping chunks."""
    metadata = doc.metadata
    article_number = str(metadata.get("article_number") or "")
    article_title = str(metadata.get("article_title") or metadata.get("section") or "")
    prefix = normalize_text(doc.page_content)[:240]
    return "|".join((source_key(doc), article_number, normalize_text(article_title), prefix))


def diversity_stats(docs: list[Document]) -> dict[str, float | int]:
    """Summarize duplicate pressure and source spread for retrieval traces."""
    if not docs:
        return {
            "candidate_count": 0,
            "unique_doc_id_count": 0,
            "unique_source_count": 0,
            "unique_parent_count": 0,
            "doc_duplicate_rate": 0.0,
            "source_duplicate_rate": 0.0,
        }

    doc_ids = [str(doc.metadata.get("doc_id") or "") for doc in docs if str(doc.metadata.get("doc_id") or "")]
    source_keys = [source_key(doc) for doc in docs if source_key(doc)]
    parent_ids = [str(doc.metadata.get("parent_id") or "") for doc in docs if str(doc.metadata.get("parent_id") or "")]
    unique_doc_ids = len(set(doc_ids))
    unique_sources = len(set(source_keys))
    unique_parents = len(set(parent_ids))
    total = len(docs)
    return {
        "candidate_count": total,
        "unique_doc_id_count": unique_doc_ids,
        "unique_source_count": unique_sources,
        "unique_parent_count": unique_parents,
        "doc_duplicate_rate": max(0.0, 1.0 - (unique_doc_ids / total)) if doc_ids else 0.0,
        "source_duplicate_rate": max(0.0, 1.0 - (unique_sources / total)) if source_keys else 0.0,
    }


def unique_by_chunk(docs: list[Document]) -> list[Document]:
    """Deduplicate candidates by chunk key while preserving original order."""
    unique_docs: list[Document] = []
    seen: set[str] = set()
    for doc in docs:
        chunk_key = str(doc.metadata.get("chunk_id") or doc.metadata.get("parent_id") or "")
        if not chunk_key or chunk_key in seen:
            continue
        seen.add(chunk_key)
        unique_docs.append(doc)
    return unique_docs


def deduplicate_near_duplicate_chunks(docs: list[Document]) -> list[Document]:
    """Drop highly overlapping chunk variants before expensive reranking."""
    unique_docs: list[Document] = []
    seen: set[str] = set()
    for doc in docs:
        fingerprint = duplicate_fingerprint(doc)
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_docs.append(doc)
    return unique_docs


def select_pre_rerank_candidates(
    docs: list[Document],
    limit: int,
    score_fn: Callable[[Document], float],
) -> list[Document]:
    """Select a more diverse candidate pool before reranking."""
    remaining = [(doc, score_fn(doc)) for doc in docs]
    selected: list[Document] = []
    source_counts: dict[str, int] = {}
    parent_counts: dict[str, int] = {}

    while remaining and len(selected) < limit:
        best_index = 0
        best_adjusted_score = float("-inf")
        for index, (doc, base_score) in enumerate(remaining):
            source = source_key(doc)
            parent_id = str(doc.metadata.get("parent_id") or "")
            adjusted_score = base_score
            if source:
                adjusted_score -= source_counts.get(source, 0) * 1.4
                if source_counts.get(source, 0) == 0:
                    adjusted_score += 0.2
            if parent_id:
                adjusted_score -= parent_counts.get(parent_id, 0) * 2.0
                if parent_counts.get(parent_id, 0) == 0:
                    adjusted_score += 0.1
            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_index = index

        doc, _ = remaining.pop(best_index)
        selected.append(doc)
        source = source_key(doc)
        parent_id = str(doc.metadata.get("parent_id") or "")
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
        if parent_id:
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
    return selected


def aggregate_child_scores(scores: list[float]) -> float:
    """Aggregate multiple supporting child chunks with diminishing returns."""
    total = 0.0
    for index, score in enumerate(sorted(scores, reverse=True)):
        total += score / (index + 1)
    return total


def select_diverse_parents(
    ranked_parents: list[tuple[Document, float]],
    limit: int,
) -> tuple[list[Document], list[float]]:
    """Select top parents with soft source penalties instead of hard caps."""
    selected_docs: list[Document] = []
    selected_scores: list[float] = []
    remaining = list(ranked_parents)
    per_source_count: dict[str, int] = {}
    per_doc_count: dict[str, int] = {}

    while remaining and len(selected_docs) < limit:
        best_index = 0
        best_adjusted_score = float("-inf")
        for index, (doc, score) in enumerate(remaining):
            source = source_key(doc)
            doc_id = str(doc.metadata.get("doc_id") or "")
            adjusted_score = score
            if source:
                adjusted_score -= per_source_count.get(source, 0) * 1.2
                if per_source_count.get(source, 0) == 0:
                    adjusted_score += 0.25
            if doc_id:
                adjusted_score -= per_doc_count.get(doc_id, 0) * 0.6
            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_index = index
        doc, score = remaining.pop(best_index)
        selected_docs.append(doc)
        selected_scores.append(score)
        source = source_key(doc)
        doc_id = str(doc.metadata.get("doc_id") or "")
        if source:
            per_source_count[source] = per_source_count.get(source, 0) + 1
        if doc_id:
            per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1
    return selected_docs, selected_scores
