"""Unified research store (stage 3): both collection windows in one schema.

  build_store        merge the frozen window-1 database and the window-2 CSV exports
                     into one de-duplicated parquet store
  field_reliability  audit every stored field: null / zero rates and the measured
                     refresh cadence, before any of it is used as a feature

Paths are relative to the repo root and can be overridden by environment variables.
The data itself is private and not part of this repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
W1_DB = Path(os.environ.get("WR_WINDOW1_DB") or (REPO_ROOT / "data" / "frozen_export" / "wallet_research.db"))
W2_ROOT = Path(os.environ.get("WR_WINDOW2_DIR") or (REPO_ROOT / "data" / "exports"))
STORE_DIR = Path(os.environ.get("WR_STORE_DIR") or (REPO_ROOT / "data" / "store"))


def wallet_labels() -> dict[str, str]:
    """address -> watchlist label (a pseudonym such as WALLET_01). Read from the
    private config/wallets.json; empty if it is absent. Reports are keyed by label
    so that no address, or fragment of one, is ever written to an output file."""
    path = REPO_ROOT / "config" / "wallets.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {w["address"]: w.get("label") or f"wallet_{i}"
            for i, w in enumerate(data.get("target_wallets") or [], 1) if w.get("address")}


def label_of(addr: str, labels: dict[str, str]) -> str:
    return labels.get(addr, "unlisted")
