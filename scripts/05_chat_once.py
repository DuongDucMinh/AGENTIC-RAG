"""Run one LangGraph chat request from the command line."""

import argparse
import json

from app.core.logging import configure_logging
from app.services.chat_service import ChatService


# Goi LangGraph agent mot lan tu CLI.
def main() -> None:
    """Invoke the agent once and print answer, citations, and debug trace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--session-id", default="cli-demo")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging()
    result = ChatService().chat(session_id=args.session_id, question=args.question, debug=args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
