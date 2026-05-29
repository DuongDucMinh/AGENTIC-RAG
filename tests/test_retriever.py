"""Unit tests for retrieval diversity and aggregation helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from app.retrieval.retriever import _detect_query_intent, _is_domain_query
from app.retrieval.qdrant_store import (
    clear_qdrant_caches,
    get_qdrant_client,
    get_vector_store,
)
from app.retrieval.ranking import (
    aggregate_child_scores,
    deduplicate_near_duplicate_chunks,
    select_diverse_parents,
    select_pre_rerank_candidates,
)


def _doc(
    *,
    text: str,
    source: str,
    parent_id: str,
    doc_id: str,
    article_number: str = "1",
    article_title: str = "Điều 1",
    rrf_score: float = 0.0,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "so_ky_hieu": source,
            "doc_id": doc_id,
            "parent_id": parent_id,
            "article_number": article_number,
            "article_title": article_title,
            "rrf_score": rrf_score,
        },
    )


def test_deduplicate_near_duplicate_chunks_removes_overlapping_variants():
    duplicate_text = "Điều 1 mức thu lệ phí trước bạ với xe máy áp dụng như sau. " * 5
    docs = [
        _doc(
            text=duplicate_text,
            source="A",
            parent_id="p1",
            doc_id="doc-a",
        ),
        _doc(
            text=duplicate_text.replace(" ", "  "),
            source="A",
            parent_id="p2",
            doc_id="doc-a",
        ),
        _doc(
            text="Điều 2 đối tượng chịu lệ phí trước bạ gồm các tài sản sau. " * 4,
            source="A",
            parent_id="p3",
            doc_id="doc-a",
            article_number="2",
            article_title="Điều 2",
        ),
    ]

    deduped = deduplicate_near_duplicate_chunks(docs)

    assert len(deduped) == 2
    assert deduped[0].metadata["parent_id"] == "p1"
    assert deduped[1].metadata["parent_id"] == "p3"


def test_select_pre_rerank_candidates_reduces_single_source_domination():
    docs = [
        _doc(text="A1", source="97/2015/QH13", parent_id="p1", doc_id="doc-a", rrf_score=10.0),
        _doc(text="A2", source="97/2015/QH13", parent_id="p2", doc_id="doc-a", rrf_score=9.5),
        _doc(text="B1", source="45/2011/NĐ-CP", parent_id="p3", doc_id="doc-b", rrf_score=9.4),
    ]

    selected = select_pre_rerank_candidates(
        docs,
        limit=2,
        score_fn=lambda doc: float(doc.metadata["rrf_score"]),
    )

    assert [doc.metadata["parent_id"] for doc in selected] == ["p1", "p3"]


def test_select_diverse_parents_uses_soft_penalty_not_hard_cap():
    ranked_parents = [
        (_doc(text="A1", source="97/2015/QH13", parent_id="p1", doc_id="doc-a"), 10.0),
        (_doc(text="A2", source="97/2015/QH13", parent_id="p2", doc_id="doc-a"), 9.0),
        (_doc(text="B1", source="45/2011/NĐ-CP", parent_id="p3", doc_id="doc-b"), 8.5),
    ]

    selected_docs, selected_scores = select_diverse_parents(ranked_parents, limit=3)

    assert [doc.metadata["parent_id"] for doc in selected_docs] == ["p1", "p3", "p2"]
    assert selected_scores == [10.0, 8.5, 9.0]


def test_aggregate_child_scores_applies_diminishing_returns():
    assert aggregate_child_scores([5.0, 4.0, 3.0]) == 8.0


def test_is_domain_query_rejects_traffic_violation_false_positive():
    assert _is_domain_query("mức phạt vi phạm giao thông xe máy là bao nhiêu?") is False


def test_is_domain_query_keeps_legit_registration_fee_question():
    assert _is_domain_query("mức thu lệ phí trước bạ xe máy là bao nhiêu?") is True


def test_rate_question_with_chiu_is_muc_thu_not_doi_tuong():
    assert _detect_query_intent("Nhà và đất chịu lệ phí trước bạ theo tỷ lệ bao nhiêu?") == "muc_thu"
    assert _detect_query_intent("Súng săn chịu lệ phí trước bạ theo mức nào?") == "muc_thu"


def test_object_question_stays_doi_tuong():
    assert _detect_query_intent("đối tượng chịu lệ phí trước bạ gồm những gì?") == "doi_tuong"


def test_can_cu_tinh_intent_wins_over_generic_gom_terms():
    assert _detect_query_intent("Căn cứ để tính lệ phí trước bạ gồm những thành phần chính nào?") == "can_cu_tinh"
    assert _detect_query_intent("Giá trị đất dùng để tính lệ phí trước bạ được xác định theo cách nào?") == "can_cu_tinh"


def test_tham_quyen_intent_for_who_decides_rate():
    assert _detect_query_intent("Ai quyết định mức cụ thể đối với ô tô chở người dưới 10 chỗ?") == "tham_quyen"


def test_qdrant_client_is_cached_per_process():
    clear_qdrant_caches()
    fake_settings = SimpleNamespace(
        qdrant_url="http://unit-test-qdrant:6333",
        qdrant_timeout_s=7,
        qdrant_collection="legal_tax_child_chunks",
        use_qdrant_sparse=False,
    )

    with patch("app.retrieval.qdrant_store.get_settings", return_value=fake_settings), patch(
        "app.retrieval.qdrant_store.QdrantClient",
        side_effect=lambda **kwargs: {"client_kwargs": kwargs},
    ) as client_ctor:
        first = get_qdrant_client()
        second = get_qdrant_client()

    assert first is second
    assert client_ctor.call_count == 1


def test_vector_store_is_cached_until_reset():
    clear_qdrant_caches()
    fake_settings = SimpleNamespace(
        qdrant_url="http://unit-test-qdrant:6333",
        qdrant_timeout_s=7,
        qdrant_collection="legal_tax_child_chunks",
        use_qdrant_sparse=False,
    )
    fake_client = SimpleNamespace(collection_exists=lambda _name: True)

    with patch("app.retrieval.qdrant_store.get_settings", return_value=fake_settings), patch(
        "app.retrieval.qdrant_store._get_qdrant_client_cached",
        return_value=fake_client,
    ), patch(
        "app.retrieval.qdrant_store.get_dense_embeddings",
        return_value="dense-embeddings",
    ), patch(
        "app.retrieval.qdrant_store.QdrantVectorStore",
        side_effect=lambda **kwargs: {"vector_store_kwargs": kwargs},
    ) as store_ctor:
        first = get_vector_store()
        second = get_vector_store()
        clear_qdrant_caches()
        third = get_vector_store()

    assert first is second
    assert third is not first
    assert store_ctor.call_count == 2
