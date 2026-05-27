"""Logging setup used by the API and command-line checkpoint scripts."""

import logging
import sys

from app.core.config import get_settings


# Cau hinh logging format thong nhat cho API va script CLI.
def configure_logging() -> None:
    """Configure a consistent console log format for all project modules."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
