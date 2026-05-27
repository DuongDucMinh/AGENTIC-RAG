"""Run retrieval on a small set of representative tax/legal questions."""

import json

from app.core.logging import configure_logging
from app.retrieval.retriever import LegalRetriever

BENCHMARK_QUERIES = [
    "muc thu le phi truoc ba duoc quy dinh nhu the nao?",
    "doi tuong chiu le phi truoc ba gom nhung gi?",
    "nguyen tac xac dinh muc thu phi va le phi la gi?",
    "to chuc thu phi, le phi co trach nhiem gi?",
    "nguyen tac xac dinh muc thu le phi truoc ba la gi?",
    "trach nhiem cua to chuc thu le phi truoc ba la gi?",
    "quy dinh ve xay dung nha o la gi?",
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
