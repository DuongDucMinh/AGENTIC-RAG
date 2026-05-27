"""Run custom keyword smoke evaluation against local retrieval and chat APIs."""

import argparse
import json
import time
from pathlib import Path

import httpx

from evals.ragas_lite import score_case


# Parse tham so CLI cho evaluation lite.
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for custom keyword smoke evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=Path("evals/chat_eval_cases.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_reports"))
    return parser.parse_args()


# Doc file JSONL cau hoi evaluation.
def load_cases(path: Path, limit: int = 0) -> list[dict]:
    """Load evaluation cases from JSONL."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


# Ghi summary markdown ngan gon de xem nhanh ket qua.
def write_summary(path: Path, results: list[dict]) -> None:
    """Write a Markdown summary of custom keyword smoke scores."""
    total = len(results)
    keys = ["retrieval_hit", "citation_present", "answer_relevance_lite", "refusal_quality"]
    lines = ["# Custom Keyword Smoke Eval Summary", "", "This is not a full RAGAS replacement; use `python -m evals.run_eval` as the main chat gate.", ""]
    for key in keys:
        passed = sum(1 for row in results if row["scores"].get(key))
        lines.append(f"- {key}: {passed}/{total}")
    avg_precision = sum(row["scores"].get("context_precision_lite", 0.0) for row in results) / max(total, 1)
    lines.append(f"- context_precision_lite_avg: {avg_precision:.3f}")
    path.write_text("\n".join(lines), encoding="utf-8")


# Chay retrieval va chat API cho tung cau hoi roi tinh metric lite.
def main() -> None:
    """Evaluate local APIs and write JSONL plus Markdown reports."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.dataset, args.limit)
    results: list[dict] = []

    with httpx.Client(timeout=180) as client:
        for index, case in enumerate(cases, start=1):
            start = time.perf_counter()
            try:
                retrieval_response = client.post(f"{args.base_url}/retrieval/search", json={"query": case["question"], "top_k": 5, "debug": True})
                chat_response = client.post(f"{args.base_url}/chat", json={"session_id": f"keyword-smoke-{index}", "question": case["question"], "debug": True})
                retrieval_response.raise_for_status()
                chat_response.raise_for_status()
                retrieval = retrieval_response.json()
                chat = chat_response.json()
            except httpx.HTTPError as exc:
                latency_ms = (time.perf_counter() - start) * 1000
                results.append({"case": case, "scores": {}, "latency_ms": latency_ms, "error": f"{type(exc).__name__}: {exc}"})
                continue
            latency_ms = (time.perf_counter() - start) * 1000
            scores = score_case(case, retrieval.get("contexts", []), chat.get("answer", ""), chat.get("citations", []))
            results.append({"case": case, "scores": scores, "latency_ms": latency_ms, "retrieval_trace": retrieval.get("trace", {})})

    output_jsonl = args.output_dir / "ragas_lite_results.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(args.output_dir / "ragas_lite_summary.md", results)
    print(f"wrote={output_jsonl}")
    print(f"summary={args.output_dir / 'ragas_lite_summary.md'}")


if __name__ == "__main__":
    main()
