"""Module A — live on-chain feature engineering.

Computes the 14 derived REAL/INT columns from the difference between the current
token state and the token's previously committed snapshot. These velocities/deltas
only exist in the live snapshot continuity and cannot be reconstructed after the fact.

compute() is called from two places:
  - the snapshot archive loop, with commit=True -> updates the per-token baseline
  - the event labeler at trade time, with commit=False -> exact-moment features
    without disturbing the archive's delta baseline

Missing/None inputs (e.g. the first snapshot, or a field absent from a poll)
yield None for the affected derived value rather than a wrong number.

Lesson recorded here because it shaped the analysis stage: the underlying fields
refresh at the REST polling cadence, which is slower and less regular than the
snapshot cadence. Differencing them snapshot-to-snapshot therefore produces mostly
zeros punctuated by jumps. The analysis package treats these columns as degenerate
and rebuilds deltas over windows matched to each field's measured refresh cadence
(see analysis/config.py and analysis/features.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The 14 derived columns this module owns (must match schema._DERIVED_DDL).
DERIVED_COLUMNS = (
    "holder_delta_snap", "holder_velocity_1m",
    "liquidity_delta_snap", "liquidity_velocity_1m",
    "bundler_pct_delta", "sniper_pct_delta",
    "insider_pct_delta", "dev_pct_delta",
    "buy_sell_ratio_1m", "buy_sell_ratio_5m",
    "volume_acceleration_1m", "progress_velocity",
    "sol_reserves_delta", "top10_concentration_delta",
)


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _int_str(v: Any) -> int | None:
    """Parse a reserve value stored as an integer string (lamports/raw units)."""
    if v is None:
        return None
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def _ratio(buy: Any, sell: Any) -> float | None:
    b, s = _f(buy), _f(sell)
    if b is None or s is None or s <= 0:
        return None
    return b / s


@dataclass
class _State:
    ts: int
    holder: float | None
    liquidity: float | None
    bundler: float | None
    sniper: float | None
    insider: float | None
    dev: float | None
    volume_1m: float | None
    progress: float | None
    sol_reserves: int | None
    top10: float | None


@dataclass
class FeatureEngine:
    _state: dict[str, _State] = field(default_factory=dict)

    def compute(self, token: str, snap: dict[str, Any], ts: int, commit: bool = True) -> dict[str, Any]:
        cur = _State(
            ts=ts,
            holder=_f(snap.get("holder")),
            liquidity=_f(snap.get("liquidity")),
            bundler=_f(snap.get("bundler_held_percentage")),
            sniper=_f(snap.get("sniper_held_percentage")),
            insider=_f(snap.get("insider_held_percentage")),
            dev=_f(snap.get("dev_held_percentage")),
            volume_1m=_f(snap.get("volume_1m_usd")),
            progress=_f(snap.get("progress_percent")),
            sol_reserves=_int_str(snap.get("real_sol_reserves")),
            top10=_f(snap.get("top10_holder_percentage")),
        )
        prev = self._state.get(token)

        out: dict[str, Any] = {c: None for c in DERIVED_COLUMNS}
        # ratios are point-in-time (no prev needed)
        out["buy_sell_ratio_1m"] = _ratio(snap.get("buy_1m"), snap.get("sell_1m"))
        out["buy_sell_ratio_5m"] = _ratio(snap.get("buy_5m"), snap.get("sell_5m"))

        if prev is not None:
            dt = ts - prev.ts
            per_min = 60.0 / dt if dt > 0 else None

            def delta(a, b):
                return None if (a is None or b is None) else a - b

            out["holder_delta_snap"] = (
                None if (cur.holder is None or prev.holder is None)
                else int(round(cur.holder - prev.holder)))
            out["liquidity_delta_snap"] = delta(cur.liquidity, prev.liquidity)
            out["bundler_pct_delta"] = delta(cur.bundler, prev.bundler)
            out["sniper_pct_delta"] = delta(cur.sniper, prev.sniper)
            out["insider_pct_delta"] = delta(cur.insider, prev.insider)
            out["dev_pct_delta"] = delta(cur.dev, prev.dev)
            out["top10_concentration_delta"] = delta(cur.top10, prev.top10)
            out["sol_reserves_delta"] = (
                None if (cur.sol_reserves is None or prev.sol_reserves is None)
                else cur.sol_reserves - prev.sol_reserves)

            if per_min is not None:
                hd = delta(cur.holder, prev.holder)
                ld = delta(cur.liquidity, prev.liquidity)
                vd = delta(cur.volume_1m, prev.volume_1m)
                pd = delta(cur.progress, prev.progress)
                out["holder_velocity_1m"] = None if hd is None else hd * per_min
                out["liquidity_velocity_1m"] = None if ld is None else ld * per_min
                out["volume_acceleration_1m"] = None if vd is None else vd * per_min
                out["progress_velocity"] = None if pd is None else pd * per_min

        if commit:
            self._state[token] = cur
        return out

    def reset(self, token: str) -> None:
        """Drop a token's baseline when its subscription ends."""
        self._state.pop(token, None)
