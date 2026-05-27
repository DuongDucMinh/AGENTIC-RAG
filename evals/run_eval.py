"""Chat-level evaluation runner for manual regression checks against /chat."""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _load_cases() -> list[dict[str, Any]]:
    """Load the chat-level evaluation cases."""
    path = Path(__file__).with_name("chat_eval_cases.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--start", type=int, default=1, help="1-based case index to start from")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("eval_reports/chat_eval_results.jsonl"))
    return parser.parse_args()


def _citation_matches(citation: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return True when one citation matches expected metadata fields."""
    if expected.get("doc_id") and str(citation.get("doc_id")) != str(expected["doc_id"]):
        return False
    if expected.get("so_ky_hieu") and str(citation.get("so_ky_hieu")) != str(expected["so_ky_hieu"]):
        return False
    if "article_number" in expected and expected.get("article_number") != citation.get("article_number"):
        return False
    title_contains = expected.get("title_contains")
    if title_contains and title_contains.lower() not in str(citation.get("title", "")).lower():
        return False
    return True


def _citation_hit(citations: list[dict[str, Any]], expected_citations: list[dict[str, Any]]) -> bool:
    """Return True when at least one expected citation matches an actual citation."""
    if not expected_citations:
        return True
    return any(_citation_matches(citation, expected) for citation in citations for expected in expected_citations)


def _keywords_hit(answer: str, keywords: list[str]) -> bool:
    """Return True when all expected keywords appear in the answer."""
    lowered = answer.lower()
    return all(keyword.lower() in lowered for keyword in keywords)


def _forbidden_keywords_absent(answer: str, keywords: list[str]) -> bool:
    """Return True when forbidden keywords do not appear in the answer."""
    lowered = answer.lower()
    return all(keyword.lower() not in lowered for keyword in keywords)


def _mode_quality_ok(answer: str, expected_mode: str) -> bool:
    """Check minimal answer behavior for non-grounded modes."""
    lowered = answer.lower()
    if expected_mode == "clarification":
        return "?" in answer or any(marker in lowered for marker in ("cụ thể", "trường hợp", "loại phí", "nội dung", "thông tin"))
    if expected_mode == "out_of_domain":
        return any(marker in lowered for marker in ("chỉ hỗ trợ", "ngoài phạm vi", "thuế", "phí", "lệ phí"))
    if expected_mode == "insufficient_context":
        return any(marker in lowered for marker in ("chưa có đủ căn cứ", "không đủ căn cứ", "dữ liệu đã index"))
    return True


def _evaluate_case(case: dict[str, Any], payload: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    """Evaluate a single /chat payload against an eval case."""
    answer = str(payload.get("answer", ""))
    citations = payload.get("citations", [])
    trace = payload.get("retrieval_trace") or {}
    response_mode = trace.get("response_mode")
    out_of_domain = bool(payload.get("out_of_domain"))

    expected_mode = case.get("expected_response_mode") or case.get("expected_mode")
    expected_out_of_domain = bool(case.get("expected_out_of_domain", expected_mode == "out_of_domain"))
    mode_ok = response_mode == expected_mode or (expected_mode == "out_of_domain" and out_of_domain)
    out_of_domain_ok = out_of_domain == expected_out_of_domain
    keywords_ok = _keywords_hit(answer, case.get("expected_keywords", []))
    forbidden_ok = _forbidden_keywords_absent(answer, case.get("forbidden_keywords", []))
    mode_quality_ok = _mode_quality_ok(answer, expected_mode)
    citation_ok = _citation_hit(citations, case.get("expected_citations", []))
    empty_citations_ok = True
    if expected_mode in {"clarification", "out_of_domain", "insufficient_context"}:
        empty_citations_ok = not citations

    passed = mode_ok and out_of_domain_ok and keywords_ok and forbidden_ok and mode_quality_ok and citation_ok and empty_citations_ok
    return passed, {
        "mode_ok": mode_ok,
        "out_of_domain_ok": out_of_domain_ok,
        "keywords_ok": keywords_ok,
        "forbidden_ok": forbidden_ok,
        "mode_quality_ok": mode_quality_ok,
        "citation_ok": citation_ok,
        "empty_citations_ok": empty_citations_ok,
    }


def main() -> None:
    """Run chat eval cases against a local FastAPI server."""
    args = _parse_args()
    base_url = args.base_url.rstrip("/")
    cases = _load_cases()
    if args.start > 1:
        cases = cases[args.start - 1 :]
    if args.limit:
        cases = cases[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    per_mode = Counter()
    rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    try:
        with httpx.Client(timeout=180) as client:
            for index, case in enumerate(cases, start=args.start):
                start = time.perf_counter()
                try:
                    response = client.post(
                        f"{base_url}/chat",
                        json={"session_id": f"eval-{index}", "question": case["question"], "debug": True},
                    )
                    latency_ms = (time.perf_counter() - start) * 1000
                    ok = response.status_code == 200
                    payload = response.json() if ok else {"error": response.text}
                except httpx.HTTPError as exc:
                    latency_ms = (time.perf_counter() - start) * 1000
                    ok = False
                    payload = {"error": f"{type(exc).__name__}: {exc}"}
                case_passed = False
                checks: dict[str, bool] = {}
                if ok:
                    case_passed, checks = _evaluate_case(case, payload)
                    if case_passed:
                        per_mode[case.get("expected_response_mode") or case.get("expected_mode")] += 1
                passed += int(case_passed)
                row = {
                    "case_index": index,
                    "question": case["question"],
                    "passed": case_passed,
                    "latency_ms": latency_ms,
                    "checks": checks,
                    "expected_response_mode": case.get("expected_response_mode") or case.get("expected_mode"),
                    "actual_response_mode": (payload.get("retrieval_trace") or {}).get("response_mode") if ok else None,
                    "out_of_domain": payload.get("out_of_domain") if ok else None,
                    "citations": payload.get("citations", []) if ok else [],
                    "error": payload.get("error") if not ok else "",
                }
                rows.append(row)
                logger.info(
                    "case=%s passed=%s latency_ms=%.2f expected_mode=%s question=%s checks=%s",
                    index,
                    case_passed,
                    latency_ms,
                    case.get("expected_response_mode") or case.get("expected_mode"),
                    case["question"],
                    checks,
                )
                if not case_passed:
                    logger.info("failure_payload=%s", json.dumps(payload, ensure_ascii=False)[:1600])
    except httpx.ConnectError:
        logger.error("Could not connect to %s/chat. Start the API first with: uvicorn app.main:app --reload", base_url)
        return
    logger.info(
        "eval_completed passed=%s total=%s total_latency_ms=%.2f per_mode=%s",
        passed,
        len(cases),
        (time.perf_counter() - start_all) * 1000,
        dict(per_mode),
    )
    with args.output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("eval_report=%s", args.output)


if __name__ == "__main__":
    main()
