"""Small API-based evaluation runner for manual regression checks."""

import json
import logging
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Chay bo golden questions co ban qua API /chat.
def main() -> None:
    """Run golden questions against a local FastAPI server."""
    base_url = "http://127.0.0.1:8000"
    path = Path(__file__).with_name("golden_questions.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = 0
    start_all = time.perf_counter()
    with httpx.Client(timeout=120) as client:
        for index, row in enumerate(rows, start=1):
            start = time.perf_counter()
            response = client.post(
                f"{base_url}/chat",
                json={"session_id": f"eval-{index}", "question": row["question"], "debug": True},
            )
            latency_ms = (time.perf_counter() - start) * 1000
            ok = response.status_code == 200
            payload = response.json() if ok else {"error": response.text}
            answer = payload.get("answer", "")
            keyword_hit = all(keyword.lower() in answer.lower() for keyword in row.get("expected_keywords", []))
            citation_ok = bool(payload.get("citations")) if row["type"] == "in_domain" else True
            case_passed = ok and keyword_hit and citation_ok
            passed += int(case_passed)
            logger.info("case=%s passed=%s latency_ms=%.2f question=%s", index, case_passed, latency_ms, row["question"])
            if not case_passed:
                logger.info("failure_payload=%s", json.dumps(payload, ensure_ascii=False)[:1200])
    logger.info("eval_completed passed=%s total=%s total_latency_ms=%.2f", passed, len(rows), (time.perf_counter() - start_all) * 1000)


if __name__ == "__main__":
    main()
