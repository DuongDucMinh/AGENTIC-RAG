"""Run hybrid search and reranking for one query."""

import argparse
import json

from app.core.logging import configure_logging
from app.retrieval.retriever import LegalRetriever


# Chay retrieval cho mot query va in JSON ket qua.
def main() -> None:
    """Search indexed child chunks and print parent contexts/citations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    configure_logging()
    result = LegalRetriever().search(args.query, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
