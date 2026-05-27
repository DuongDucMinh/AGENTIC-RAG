"""Preview metadata filtering without indexing any document."""

import argparse
import logging

from app.core.logging import configure_logging
from app.ingestion.hf_loader import preview_selection


# Chay preview metadata filter va in sample ra console.
def main() -> None:
    """Load metadata, apply the tax-domain filter, and print a small sample."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    configure_logging()
    total, selected, sample = preview_selection(limit=args.limit)
    print(f"total_metadata_rows={total}")
    print(f"selected_count={len(selected)}")
    for index, row in enumerate(sample, start=1):
        print(
            f"sample[{index}] id={row.get('id')} "
            f"type={row.get('loai_van_ban')} "
            f"authority={row.get('co_quan_ban_hanh')} "
            f"status={row.get('tinh_trang_hieu_luc')} "
            f"title={row.get('title')}"
        )


if __name__ == "__main__":
    main()
