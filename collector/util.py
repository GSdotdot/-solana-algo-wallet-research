"""Small shared helpers: time + logging. No project dependencies."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone


def now_unix() -> int:
    return int(time.time())


def utc_day_start(ts: int) -> int:
    """Unix seconds at 00:00:00 UTC of the day containing `ts`."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp())


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
