"""Paths, wallet metadata, and the trustworthy feature set for the decoder.

The feature set deliberately EXCLUDES the collector's live velocity/delta columns:
they are degenerate (differenced snapshot-to-snapshot off fields that refresh on a
slower, non-uniform polling cadence), so we decode from RAW on-chain state only and
rebuild deltas over cadence-matched windows. Point-in-time ratios
(buy_sell_ratio_*) are fine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from common.params import is_redacted, param

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
# The frozen export (database + the watchlist used for that run). Private; not in the repo.
EXPORT_DIR = Path(os.environ.get("WR_EXPORT_DIR") or (PROJECT_ROOT / "data" / "frozen_export"))
DB_PATH = EXPORT_DIR / "wallet_research.db"
OUTPUT_DIR = ANALYSIS_DIR / "outputs"       # gitignored: results are not published


def P(key: str) -> Any:
    """Analysis parameter lookup: P("entry.model") == param("analysis.entry.model")."""
    return param(f"analysis.{key}")


def _labels(key: str) -> frozenset[str]:
    """A wallet-label list from config -> frozenset (empty when redacted)."""
    v = P(key)
    return frozenset() if (v is None or is_redacted(v)) else frozenset(v)


# --- wallets ---------------------------------------------------------------
def _wallet_labels() -> dict[str, str]:
    """address -> label from the watchlist frozen with the export. Empty when the
    private export is not present (as in the public copy)."""
    path = EXPORT_DIR / "config" / "wallets.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {w["address"]: w["label"] for w in data["target_wallets"]}


WALLET_LABELS = _wallet_labels()                       # address -> label (pseudonym)
# Which wallets fall in each group is a research finding; the lists live in the
# private config (labels such as WALLET_03).
B_WALLET_LABELS = _labels("wallets.fast_scalpers")     # shortest-hold wallets (have 1s price bars)
DORMANT = _labels("wallets.dormant")                   # no trades in the window: excluded
# Confidence tiers from the data-quality review: deep > moderate sample depth.
TIER_DEEP = _labels("wallets.tier_deep")
TIER_MODERATE = _labels("wallets.tier_moderate")

# --- feature groups (RAW, trustworthy) -------------------------------------
SCALAR_FEATURES = [
    "price", "market_cap", "fdv", "liquidity", "holder", "top10_holder_percentage",
    "total_supply", "circulating_supply", "progress_percent", "graduated",
]
TRADER_TYPE_FEATURES = [
    "sniper_count", "bundler_count", "insider_count", "dev_count",
    "sniper_held_percentage", "bundler_held_percentage",
    "insider_held_percentage", "dev_held_percentage",
]
RESERVE_FEATURES = ["real_sol_reserves", "real_token_reserves", "virtual_token_reserves"]
RATIO_FEATURES = ["buy_sell_ratio_1m", "buy_sell_ratio_5m"]   # point-in-time, OK
DERIVED_FEATURES = ["token_age_sec"]                          # computed in data.load_events

# DEGENERATE live-delta columns: never use as features (known bad).
DEGENERATE = frozenset({
    "holder_delta_snap", "holder_velocity_1m", "liquidity_delta_snap", "liquidity_velocity_1m",
    "bundler_pct_delta", "sniper_pct_delta", "insider_pct_delta", "dev_pct_delta",
    "volume_acceleration_1m", "progress_velocity", "sol_reserves_delta",
    "top10_concentration_delta",
})
# OHLCV snapshot columns are deprecated/NULL; `price` is the series.
DEPRECATED = frozenset({"ohlcv_open", "ohlcv_high", "ohlcv_low", "ohlcv_close", "ohlcv_volume"})

WINDOWS = ("1m", "5m", "30m", "1h", "2h", "4h", "8h", "24h")


def window_features(columns) -> list[str]:
    """Per-window activity columns present in `columns` (buy/sell counts + USD volumes),
    excluding the *_change_percent windows (often NULL/noisy)."""
    return [c for c in columns
            if c.startswith(("buy_", "sell_", "volume_"))
            and "change_percent" not in c and c not in DEPRECATED]


def feature_columns(columns) -> list[str]:
    """The full trustworthy raw-feature set present in a frame's columns."""
    base = (SCALAR_FEATURES + TRADER_TYPE_FEATURES + RESERVE_FEATURES
            + RATIO_FEATURES + DERIVED_FEATURES + window_features(columns))
    seen, out = set(), []
    for c in base:
        if c in columns and c not in DEGENERATE and c not in DEPRECATED and c not in seen:
            seen.add(c); out.append(c)
    return out


# ============================================================================
# FULL DATA-INVENTORY CATALOG  (analysis doctrine)
# Every module starts from THIS universe and re-tests it; usefulness is an
# empirical output, never an inherited subset. `analysis/features.py` builds it.
# ============================================================================
TIMEFRAMES = ("1m", "5m", "30m", "1h", "2h", "4h", "8h", "24h")
TRADER_CLASSES = ("sniper", "bundler", "insider", "dev")

CAT_STRUCTURAL = ["price", "market_cap", "fdv", "liquidity", "holder",
                  "top10_holder_percentage", "progress_percent", "graduated"]
CAT_SUPPLY_RESERVE = ["total_supply", "circulating_supply", "real_sol_reserves",
                      "real_token_reserves", "virtual_token_reserves"]
CAT_CLASS_LEVELS = ([f"{c}_count" for c in TRADER_CLASSES]
                    + [f"{c}_held_percentage" for c in TRADER_CLASSES])


def orderflow_cols() -> list[str]:
    """The FULL absolute order-flow ladder across all timeframes (counts + USD volumes +
    change-percent): the family earlier passes left almost entirely unused."""
    out = []
    for tf in TIMEFRAMES:
        out += [f"buy_{tf}", f"sell_{tf}", f"volume_buy_{tf}_usd", f"volume_sell_{tf}_usd",
                f"volume_{tf}_usd", f"volume_{tf}_change_percent"]
    return out


def all_snapshot_cols() -> list[str]:
    """Every raw token_snapshots column we ingest for feature-building (degenerate +
    deprecated excluded). The complete on-chain inventory, less the broken live velocities."""
    base = (["token_address", "timestamp"] + CAT_STRUCTURAL + CAT_SUPPLY_RESERVE
            + CAT_CLASS_LEVELS + orderflow_cols() + ["buy_sell_ratio_1m", "buy_sell_ratio_5m"])
    seen, out = set(), []
    for c in base:
        if c not in seen and c not in DEGENERATE and c not in DEPRECATED:
            seen.add(c); out.append(c)
    return out


# Field refresh cadence (sec, measured on the live data) -> the MINIMUM honest delta
# window. Computing a delta over a window shorter than this reproduces the
# degenerate-velocity trap. {field: seconds}; values redacted.
CADENCE_SEC = P("features.cadence_sec")
# Fields to difference (level -> cadence-matched delta), with the windows to emit (sec).
# {field: [window_sec, ...]}; each window is >= that field's cadence. Values redacted.
DELTA_FIELDS = P("features.delta_windows")
# Fields whose delta is emitted as a percentage change rather than a difference.
PCT_DELTA_FIELDS = ("liquidity", "market_cap", "real_sol_reserves")
# Lag (sec) for the short-window volume acceleration feature.
VOL_ACCEL_LAG_SEC = P("features.vol_accel_lag_sec")


def delta_feature_names(fields=None) -> list[str]:
    """Names of the cadence-matched delta features (`<field>_d<window>[_pct]`)."""
    names = []
    for f, wins in DELTA_FIELDS.items():
        if fields is not None and f not in fields:
            continue
        pct = f in PCT_DELTA_FIELDS
        names += [f"{f}_d{w}_pct" if pct else f"{f}_d{w}" for w in wins]
    return names
