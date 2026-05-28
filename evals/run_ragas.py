"""Run real RAGAS evaluation against local retrieval and chat APIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import (
    _Faithfulness,
    _IDBasedContextPrecision,
    _IDBasedContextRecall,
    _LLMContextPrecisionWithReference,
    _LLMContextRecall,
    _ResponseRelevancy,
)
from ragas.run_config import RunConfig

from app.core.config import get_settings
from app.llm.factory import get_ragas_llm_wrapper
from app.retrieval.embeddings import get_dense_embeddings


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RAGAS evaluation."""
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=Path("evals/ragas_eval_cases.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=settings.ragas_output_dir)
    parser.add_argument("--timeout", type=int, default=settings.ragas_run_timeout_s)
    parser.add_argument("--max-workers", type=int, default=settings.ragas_max_workers)
    return parser.parse_args()


def load_cases(path: Path, start: int = 0, limit: int = 0) -> list[dict[str, Any]]:
    """Load RAGAS evaluation cases from JSONL."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if start:
        rows = rows[start:]
    max_cases = get_settings().ragas_max_cases
    if limit:
        rows = rows[:limit]
    elif max_cases:
        rows = rows[:max_cases]
    return rows


def build_ragas_metrics():
    """Create standard RAGAS metrics plus deterministic source checks."""
    llm = get_ragas_llm_wrapper()
    embeddings = get_dense_embeddings()
    return [
        _Faithfulness(llm=llm),
        _ResponseRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        _LLMContextPrecisionWithReference(llm=llm, name="context_precision"),
        _LLMContextRecall(llm=llm),
        _IDBasedContextPrecision(name="source_precision"),
        _IDBasedContextRecall(name="source_recall"),
    ], llm, embeddings


def citation_keys(citation: dict[str, Any]) -> list[str]:
    """Return stable source keys from citation metadata."""
    keys: list[str] = []
    doc_id = str(citation.get("doc_id") or "").strip()
    article = citation.get("article_number")
    article_number = str(article).strip() if article is not None else ""
    parent_id = str(citation.get("parent_id") or "").strip()
    so_ky_hieu = str(citation.get("so_ky_hieu") or "").strip()
    if parent_id:
        keys.append(f"parent:{parent_id}")
    if doc_id and article_number:
        keys.append(f"doc_article:{doc_id}:{article_number}")
    if so_ky_hieu and article_number:
        keys.append(f"symbol_article:{so_ky_hieu}:{article_number}")
    if doc_id:
        keys.append(f"doc:{doc_id}")
    if so_ky_hieu:
        keys.append(f"symbol:{so_ky_hieu}")
    return keys


def expected_source_keys(case: dict[str, Any]) -> list[str]:
    """Build reference source ids from golden citation metadata."""
    keys: list[str] = []
    for citation in case.get("expected_citations", []):
        keys.extend(citation_keys(citation))
    return sorted(set(keys))


def retrieved_source_keys(citations: list[dict[str, Any]]) -> list[str]:
    """Build retrieved source ids from retrieval citations."""
    keys: list[str] = []
    for citation in citations:
        keys.extend(citation_keys(citation))
    return sorted(set(keys))


def fetch_case_artifacts(client: httpx.Client, base_url: str, case: dict[str, Any], index: int) -> dict[str, Any]:
    """Fetch retrieval contexts and grounded answer for a single RAGAS case."""
    start = time.perf_counter()
    retrieval_start = time.perf_counter()
    retrieval_response = client.post(
        f"{base_url}/retrieval/search",
        json={"query": case["question"], "top_k": 5, "debug": True},
    )
    retrieval_api_ms = (time.perf_counter() - retrieval_start) * 1000
    chat_start = time.perf_counter()
    chat_response = client.post(
        f"{base_url}/chat",
        json={"session_id": f"ragas-{index}", "question": case["question"], "debug": True},
    )
    chat_api_ms = (time.perf_counter() - chat_start) * 1000
    retrieval_response.raise_for_status()
    chat_response.raise_for_status()
    retrieval = retrieval_response.json()
    chat = chat_response.json()
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "question": case["question"],
        "ground_truth": case["ground_truth"],
        "reference_context_hints": case.get("reference_context_hints", []),
        "difficulty": case.get("difficulty", ""),
        "category": case.get("category", ""),
        "expected_source_ids": expected_source_keys(case),
        "retrieved_source_ids": retrieved_source_keys(retrieval.get("citations", [])),
        "retrieved_contexts": retrieval.get("contexts", []),
        "answer": chat.get("answer", ""),
        "citations": chat.get("citations", []),
        "retrieval_citations": retrieval.get("citations", []),
        "out_of_domain": chat.get("out_of_domain", False),
        "retrieval_trace": chat.get("retrieval_trace") or {},
        "retrieval_debug_trace": retrieval.get("trace") or {},
        "latency_ms": latency_ms,
        "retrieval_api_ms": retrieval_api_ms,
        "chat_api_ms": chat_api_ms,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows to disk."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flattened per-case rows to CSV."""
    columns = [
        "question",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "source_precision",
        "source_recall",
        "latency_ms",
        "retrieval_api_ms",
        "chat_api_ms",
        "response_mode",
        "out_of_domain",
        "citation_count",
        "difficulty",
        "category",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    metric_names: list[str],
    total_cases: int,
    skipped_non_grounded: int,
    failed_before_eval: int,
    ragas_judge_latency_ms: float,
) -> None:
    """Write a markdown summary of RAGAS scores."""
    total = len(rows)
    latencies = sorted(float(row.get("latency_ms", 0.0) or 0.0) for row in rows)
    chat_latencies = sorted(float(row.get("chat_api_ms", 0.0) or 0.0) for row in rows)
    retrieval_latencies = sorted(float(row.get("retrieval_api_ms", 0.0) or 0.0) for row in rows)

    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        index = math.ceil((p / 100) * len(values)) - 1
        return values[max(0, min(index, len(values) - 1))]

    lines = [
        "# RAGAS Summary",
        "",
        f"- requested_cases: {total_cases}",
        f"- evaluated_grounded_cases: {total}",
        f"- skipped_non_grounded_cases: {skipped_non_grounded}",
        f"- failed_before_eval_cases: {failed_before_eval}",
    ]
    for metric_name in metric_names:
        values = [
            float(row.get(metric_name, 0.0) or 0.0)
            for row in rows
            if row.get(metric_name) is not None and str(row.get(metric_name)).lower() != "nan"
        ]
        if values:
            avg = sum(values) / len(values)
            lines.append(f"- {metric_name}_avg: {avg:.4f} ({len(values)}/{total} valid)")
        else:
            lines.append(f"- {metric_name}_avg: unavailable (0/{total} valid)")
    slow_rows = sorted(rows, key=lambda row: float(row.get("latency_ms", 0.0) or 0.0), reverse=True)[:5]
    lines.extend(
        [
            f"- total_api_latency_avg_ms: {statistics.fmean(latencies) if latencies else 0.0:.2f}",
            f"- total_api_latency_p50_ms: {percentile(latencies, 50):.2f}",
            f"- total_api_latency_p95_ms: {percentile(latencies, 95):.2f}",
            f"- total_api_latency_max_ms: {max(latencies) if latencies else 0.0:.2f}",
            f"- retrieval_api_latency_avg_ms: {statistics.fmean(retrieval_latencies) if retrieval_latencies else 0.0:.2f}",
            f"- chat_api_latency_avg_ms: {statistics.fmean(chat_latencies) if chat_latencies else 0.0:.2f}",
            f"- ragas_judge_latency_ms: {ragas_judge_latency_ms:.2f}",
            "",
            "## Slowest Cases",
            "",
        ]
    )
    for row in slow_rows:
        lines.append(
            f"- {float(row.get('latency_ms', 0.0) or 0.0):.2f} ms total; "
            f"retrieval={float(row.get('retrieval_api_ms', 0.0) or 0.0):.2f} ms; "
            f"chat={float(row.get('chat_api_ms', 0.0) or 0.0):.2f} ms; "
            f"{str(row.get('question', ''))[:120]}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `run_eval.py` remains the main `/chat` behavior gate.",
            "- `source_precision` and `source_recall` are deterministic citation-id checks added to make retrieval quality less dependent on LLM judge leniency.",
            "- This RAGAS run scores only grounded-answer cases. Non-grounded cases should be investigated in `run_eval.py`.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run a real RAGAS evaluation using Groq judge and local API outputs."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.dataset, start=args.start, limit=args.limit)
    if not cases:
        raise SystemExit("No RAGAS cases found.")

    raw_rows: list[dict[str, Any]] = []
    successful_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    skipped_non_grounded = 0
    failed_before_eval = 0

    with httpx.Client(timeout=args.timeout) as client:
        for index, case in enumerate(cases, start=1 + args.start):
            try:
                artifact = fetch_case_artifacts(client, args.base_url, case, index)
            except httpx.HTTPError as exc:
                failed_before_eval += 1
                raw_rows.append(
                    {
                        "question": case["question"],
                        "ground_truth": case["ground_truth"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            response_mode = (artifact.get("retrieval_trace") or {}).get("response_mode", "")
            raw_row = {
                **artifact,
                "response_mode": response_mode,
                "citation_count": len(artifact["citations"]),
            }
            raw_rows.append(raw_row)
            if response_mode != "grounded_answer":
                skipped_non_grounded += 1
                continue
            successful_rows.append(raw_row)
            dataset_rows.append(
                {
                    "user_input": artifact["question"],
                    "retrieved_contexts": artifact["retrieved_contexts"],
                    "retrieved_context_ids": artifact["retrieved_source_ids"],
                    "reference_context_ids": artifact["expected_source_ids"],
                    "response": artifact["answer"],
                    "reference": artifact["ground_truth"],
                }
            )

    if not dataset_rows:
        output_jsonl = args.output_dir / "ragas_results.jsonl"
        write_jsonl(output_jsonl, raw_rows)
        raise SystemExit("All RAGAS cases failed before evaluation. Check API/Groq availability.")

    metrics, llm, embeddings = build_ragas_metrics()
    metric_names = [metric.name for metric in metrics]
    evaluation_dataset = EvaluationDataset.from_list(dataset_rows)
    ragas_judge_start = time.perf_counter()
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=args.timeout, max_workers=args.max_workers),
        show_progress=True,
        raise_exceptions=False,
        batch_size=1,
    )
    ragas_judge_latency_ms = (time.perf_counter() - ragas_judge_start) * 1000
    result_frame = result.to_pandas()

    merged_rows: list[dict[str, Any]] = []
    for index, row in enumerate(successful_rows):
        score_row = result_frame.iloc[index].to_dict()
        merged_rows.append(
            {
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "answer": row["answer"],
                "difficulty": row["difficulty"],
                "category": row["category"],
                "retrieved_contexts": row["retrieved_contexts"],
                "retrieved_source_ids": row["retrieved_source_ids"],
                "expected_source_ids": row["expected_source_ids"],
                "citations": row["citations"],
                "retrieval_citations": row["retrieval_citations"],
                "latency_ms": row["latency_ms"],
                "retrieval_api_ms": row["retrieval_api_ms"],
                "chat_api_ms": row["chat_api_ms"],
                "response_mode": row["response_mode"],
                "out_of_domain": row["out_of_domain"],
                "citation_count": row["citation_count"],
                **{metric_name: score_row.get(metric_name) for metric_name in metric_names},
            }
        )

    output_jsonl = args.output_dir / "ragas_results.jsonl"
    output_csv = args.output_dir / "ragas_results.csv"
    output_md = args.output_dir / "ragas_summary.md"
    output_all_jsonl = args.output_dir / "ragas_all_cases.jsonl"
    write_jsonl(output_jsonl, merged_rows)
    write_jsonl(output_all_jsonl, raw_rows)
    write_csv(output_csv, merged_rows)
    write_summary(
        output_md,
        merged_rows,
        metric_names,
        total_cases=len(cases),
        skipped_non_grounded=skipped_non_grounded,
        failed_before_eval=failed_before_eval,
        ragas_judge_latency_ms=ragas_judge_latency_ms,
    )
    print(f"wrote={output_jsonl}", flush=True)
    print(f"all_cases={output_all_jsonl}", flush=True)
    print(f"csv={output_csv}", flush=True)
    print(f"summary={output_md}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
