"""Index a small selected subset into Qdrant for checkpoint testing."""

import argparse

from app.core.logging import configure_logging
from app.services.indexing_service import IndexingService


# Chay indexing truc tiep voi subset nho de test Qdrant.
def main() -> None:
    """Run the indexing service directly without starting FastAPI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-documents", type=int, default=20)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    configure_logging()
    result = IndexingService().run(max_documents=args.max_documents, reset_collection=args.reset)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
