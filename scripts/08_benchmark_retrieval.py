"""Run retrieval on a small set of representative tax/legal questions."""

import json

from app.core.logging import configure_logging
from app.retrieval.retriever import LegalRetriever

BENCHMARK_QUERIES = [
    "mức thu lệ phí trước bạ được quy định thế nào?",
    "đối tượng chịu lệ phí trước bạ gồm những gì?",
    "nguyên tắc xác định mức thu phí và lệ phí là gì?",
    "tổ chức thu phí, lệ phí có trách nhiệm gì?",
    "nguyên tắc xác định mức thu lệ phí trước bạ là gì?",
    "trách nhiệm của tổ chức thu lệ phí trước bạ là gì?",
    "quy định về xây dựng nhà ở là gì?",
]


# Chay mot bo query dai dien de xem retriever co dung huong hay khong.
def main() -> None:
    """Execute retrieval for several benchmark questions and print compact results."""
    configure_logging()
    retriever = LegalRetriever()
    results: list[dict] = []

    for query in BENCHMARK_QUERIES:
        result = retriever.search(query, top_k=3)
        citations = result["citations"]
        trace = result["trace"]
        results.append(
            {
                "query": query,
                "top_citations": [
                    {
                        "title": citation.get("title"),
                        "so_ky_hieu": citation.get("so_ky_hieu"),
                        "article_number": citation.get("article_number"),
                        "article_title": citation.get("article_title"),
                    }
                    for citation in citations[:3]
                ],
                "trace": {
                    "keyword_heavy_query": trace.get("keyword_heavy_query"),
                    "query_intent": trace.get("query_intent"),
                    "out_of_domain": trace.get("out_of_domain"),
                    "out_of_domain_reason": trace.get("reason"),
                    "rerank_bypassed": trace.get("rerank_bypassed"),
                    "parent_count": trace.get("parent_count"),
                    "parent_scores": trace.get("parent_scores"),
                    "parent_top_candidates": trace.get("parent_top_candidates"),
                },
            }
        )

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
