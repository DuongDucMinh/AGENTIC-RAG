"""Run one LangGraph chat request from the command line."""

import argparse
import json
import sys

from app.core.logging import configure_logging
from app.services.chat_service import get_chat_service


# Goi LangGraph agent mot lan tu CLI.
def main() -> None:
    """Invoke the agent once and print answer, citations, and debug trace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--session-id", default="cli-demo")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = get_chat_service().chat(session_id=args.session_id, question=args.question, debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
