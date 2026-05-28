"""Evaluate retrieval source precision/recall without LLM judges."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.core.logging import configure_logging
from app.retrieval.retriever import LegalRetriever


def parse_args() -> argparse.Namespace:
    """Parse retrieval evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evals/retrieval_eval_cases.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("eval_reports/retrieval_source_eval.json"))
    parser.add_argument("--show-failures", action="store_true")
    return parser.parse_args()


def load_cases(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    """Load source-labeled retrieval cases."""
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases[:limit] if limit else cases


def _norm(value: Any) -> str:
    return str(value or "").strip()


def citation_matches(retrieved: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Strict source match; article must match when the reference specifies one."""
    expected_doc_id = _norm(expected.get("doc_id"))
    expected_symbol = _norm(expected.get("so_ky_hieu"))
    expected_article = _norm(expected.get("article_number"))
    got_doc_id = _norm(retrieved.get("doc_id"))
    got_symbol = _norm(retrieved.get("so_ky_hieu"))
    got_article = _norm(retrieved.get("article_number"))

    source_ok = bool(expected_doc_id and expected_doc_id == got_doc_id) or bool(expected_symbol and expected_symbol == got_symbol)
    if not source_ok:
        return False
    if expected_article:
        return expected_article == got_article
    return True


def score_case(citations: list[dict[str, Any]], expected_citations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute strict source metrics for a single case."""
    matched_expected_indexes: set[int] = set()
    matched_retrieved_indexes: set[int] = set()
    for retrieved_index, retrieved in enumerate(citations):
        for expected_index, expected in enumerate(expected_citations):
            if citation_matches(retrieved, expected):
                matched_expected_indexes.add(expected_index)
                matched_retrieved_indexes.add(retrieved_index)

    hit_count = len(matched_expected_indexes)
    precision = len(matched_retrieved_indexes) / len(citations) if citations else 0.0
    recall = hit_count / len(expected_citations) if expected_citations else 0.0
    hit_at_1 = bool(citations and any(citation_matches(citations[0], expected) for expected in expected_citations))
    hit_at_k = hit_count > 0
    return {
        "source_precision": precision,
        "source_recall": recall,
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "matched_expected_count": hit_count,
    }


def summarize(rows: list[dict[str, Any]], total_latency_ms: float) -> dict[str, Any]:
    """Build aggregate metrics by all cases and category."""
    latencies = [float(row["retrieval_total_ms"]) for row in rows]
    summary: dict[str, Any] = {
        "case_count": len(rows),
        "source_precision_avg": statistics.fmean(row["source_precision"] for row in rows) if rows else 0.0,
        "source_recall_avg": statistics.fmean(row["source_recall"] for row in rows) if rows else 0.0,
        "hit_at_1": statistics.fmean(1.0 if row["hit_at_1"] else 0.0 for row in rows) if rows else 0.0,
        "hit_at_k": statistics.fmean(1.0 if row["hit_at_k"] else 0.0 for row in rows) if rows else 0.0,
        "retrieval_latency_avg_ms": statistics.fmean(latencies) if latencies else 0.0,
        "retrieval_latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "retrieval_latency_max_ms": max(latencies) if latencies else 0.0,
        "wall_latency_ms": total_latency_ms,
        "by_category": {},
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("category") or "unknown")].append(row)
    for category, group_rows in sorted(groups.items()):
        summary["by_category"][category] = {
            "case_count": len(group_rows),
            "source_precision_avg": statistics.fmean(row["source_precision"] for row in group_rows),
            "source_recall_avg": statistics.fmean(row["source_recall"] for row in group_rows),
            "hit_at_1": statistics.fmean(1.0 if row["hit_at_1"] else 0.0 for row in group_rows),
            "hit_at_k": statistics.fmean(1.0 if row["hit_at_k"] else 0.0 for row in group_rows),
        }
    return summary


def main() -> None:
    """Run retrieval-only source evaluation."""
    args = parse_args()
    configure_logging()
    cases = load_cases(args.dataset, limit=args.limit)
    retriever = LegalRetriever()
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        result = retriever.search(case["question"], top_k=args.top_k)
        citations = result.get("citations", [])
        trace = result.get("trace", {})
        scores = score_case(citations, case.get("expected_citations", []))
        rows.append(
            {
                "index": index,
                "question": case["question"],
                "category": case.get("category", ""),
                "difficulty": case.get("difficulty", ""),
                "expected_citations": case.get("expected_citations", []),
                "citations": citations,
                "retrieval_total_ms": trace.get("retrieval_total_ms", 0.0),
                "vector_store_ms": trace.get("vector_store_ms", 0.0),
                "query_intent": trace.get("query_intent", ""),
                **scores,
            }
        )
    total_latency_ms = (time.perf_counter() - start) * 1000
    output = {"summary": summarize(rows, total_latency_ms), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    if args.show_failures:
        for row in rows:
            if row["source_recall"] < 1.0 or row["source_precision"] < 0.5:
                print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
