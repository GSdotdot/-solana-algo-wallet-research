"""C0 — data layer. Load the frozen export into clean analysis frames.

Two analysis units:
  events    : one row per labeled trade (ENTRY/EXIT/ADD/PARTIAL_EXIT/FLIP) with the
              on-chain context (labeled_events) joined to the trade outcome (wallet_trades)
              and token identity. THE matrix for entry-condition work.
  positions : reconstructed round-trips (open ENTRY/FLIP -> closing EXIT) per (wallet,token),
              carrying entry_* and exit_* features + hold + net PnL + size. THE matrix for
              exit-rule / sizing / hold work.

Run `python -m analysis.data` for a sanity check against the export.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import config

OPEN_LABELS = ("ENTRY", "FLIP")
CLOSE_LABELS = ("EXIT", "PARTIAL_EXIT")


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)


def load_events() -> pd.DataFrame:
    """Per-event matrix: labeled_events joined to wallet_trades (outcomes) and
    token_identity. Dormant wallets dropped; reserves coerced to numeric; token age
    derived. Wallets are referred to by their watchlist label from here on, never by
    address; a wallet missing from the watchlist gets an anonymous label (unlisted_01, ...)."""
    con = _conn()
    try:
        le = pd.read_sql("SELECT * FROM labeled_events", con)
        wt = pd.read_sql(
            "SELECT trade_id, net_pnl_usd, position_pnl_usd, prio_bribe_fee_usd, "
            "hold_duration_sec, cumulative_position_tokens, trade_sequence_on_token, "
            "time_since_last_trade_any, time_since_last_trade_same_token, "
            "other_wallets_active, dex_program FROM wallet_trades", con)
        ti = pd.read_sql(
            "SELECT token_address, token_name, token_symbol, creation_time, decimals "
            "FROM token_identity", con)
    finally:
        con.close()

    unknown = sorted(set(le["wallet_address"]) - set(config.WALLET_LABELS))
    anon = {a: f"unlisted_{i:02d}" for i, a in enumerate(unknown, 1)}
    le["wallet"] = le["wallet_address"].map({**config.WALLET_LABELS, **anon})
    le = le[~le["wallet"].isin(config.DORMANT)].copy()

    df = le.merge(wt, on="trade_id", how="left").merge(ti, on="token_address", how="left")
    for c in config.RESERVE_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["token_age_sec"] = (df["timestamp"] - df["creation_time"]).where(df["creation_time"].notna())
    df["is_open"] = df["label"].isin(OPEN_LABELS)
    df["is_close"] = df["label"].isin(CLOSE_LABELS)
    return df


def reconstruct_positions(events: pd.DataFrame) -> pd.DataFrame:
    """Pair each opening ENTRY/FLIP with its closing EXIT per (wallet, token) (LIFO).
    Carries entry_* / exit_* clean features. PARTIAL_EXIT/ADD are mid-episode -> skipped
    for the primary round-trip (handled as a sensitivity check in C3)."""
    feats = config.feature_columns(events)
    rows = []
    ev = events.sort_values(["wallet", "token_address", "timestamp", "event_id"])
    for (w, tok), g in ev.groupby(["wallet", "token_address"], sort=False):
        stack = []
        sym = g["token_symbol"].iloc[0]
        for e in g.to_dict("records"):
            lab = e["label"]
            if lab in OPEN_LABELS:
                stack.append(e)
            elif lab == "EXIT":
                opener = stack.pop() if stack else None
                rec = {"wallet": w, "token_address": tok, "token_symbol": sym,
                       "exit_time": e["timestamp"], "hold_duration_sec": e["hold_duration_sec"],
                       "net_pnl_usd": e["net_pnl_usd"], "size_usd": (opener or {}).get("volume_usd"),
                       "entry_time": (opener or {}).get("timestamp"),
                       "other_wallets_active": e.get("other_wallets_active")}
                for f in feats:
                    rec[f"entry_{f}"] = (opener or {}).get(f)
                    rec[f"exit_{f}"] = e.get(f)
                rows.append(rec)
    pos = pd.DataFrame(rows)
    if not pos.empty:
        # price move over the hold (entry->exit market price), guarding div-by-zero
        ep, xp = pos["entry_price"], pos["exit_price"]
        pos["price_change_pct"] = np.where((ep > 0) & ep.notna(), (xp - ep) / ep * 100, np.nan)
        pos["tracked"] = pos["entry_time"].notna()
    return pos


def entry_state_matrix(events: pd.DataFrame | None = None, window: int | None = None) -> pd.DataFrame:
    """ENTRY/FLIP events with on-chain state filled from the native labeled_event cache
    where present, else RECOVERED from the nearest token_snapshot within +`window` s (the
    at-trade poll lands there for first-touch entries). Lifts coverage well above the raw
    has_pre_context rate, removing the lead/follow selection bias. Adds:
      state_source : 'native' | 'recovered' | 'missing'
    Recovered state is ~poll-latency post-entry: trust the slow structural fields more
    than price / short-window volume."""
    if window is None:
        window = config.P("data.state_recovery_window_sec")     # tuned privately
    if events is None:
        events = load_events()
    ent = events[events["is_open"]].copy()
    feats = config.feature_columns(ent)
    con = _conn()
    snap_cols = {r[1] for r in con.execute("PRAGMA table_info(token_snapshots)")}
    rec_cols = [c for c in feats if c in snap_cols]            # token_age_sec stays from identity
    miss_idx = ent.index[ent["market_cap"].isna()]
    ent["state_source"] = np.where(ent["market_cap"].notna(), "native", "missing")
    sel = ",".join(rec_cols)
    q = (f"SELECT {sel} FROM token_snapshots WHERE token_address=? AND timestamp>=? "
         "AND timestamp<=? AND market_cap IS NOT NULL ORDER BY timestamp LIMIT 1")
    for idx in miss_idx:
        r = ent.loc[idx]
        row = con.execute(q, (r["token_address"], int(r["timestamp"]), int(r["timestamp"]) + window)).fetchone()
        if row:
            for c, v in zip(rec_cols, row):
                ent.at[idx, c] = v
            ent.at[idx, "state_source"] = "recovered"
    con.close()
    for c in config.RESERVE_FEATURES:                          # snapshot reserves are TEXT
        if c in ent.columns:
            ent[c] = pd.to_numeric(ent[c], errors="coerce")
    ent["covered"] = ent["market_cap"].notna()
    return ent


def load_snapshots(token: str | None = None, columns: str = "*") -> pd.DataFrame:
    con = _conn()
    try:
        if token:
            return pd.read_sql(f"SELECT {columns} FROM token_snapshots WHERE token_address=? "
                               "ORDER BY timestamp", con, params=[token])
        return pd.read_sql(f"SELECT {columns} FROM token_snapshots", con)
    finally:
        con.close()


def load_ticks(token: str | None = None) -> pd.DataFrame:
    """1-second price bars (table `price_ticks`)."""
    con = _conn()
    try:
        q = "SELECT * FROM price_ticks"
        if token:
            return pd.read_sql(q + " WHERE token_address=? ORDER BY unix_time", con, params=[token])
        return pd.read_sql(q, con)
    finally:
        con.close()


def _self_test() -> None:
    ev = load_events()
    pos = reconstruct_positions(ev)
    feats = config.feature_columns(ev)
    print(f"events: {len(ev):,} rows, {ev.shape[1]} cols")
    print(f"wallets: {sorted(ev['wallet'].unique())}")
    print(f"label mix: {ev['label'].value_counts().to_dict()}")
    print(f"\nfeature set ({len(feats)} trustworthy raw cols):")
    print(" ", feats)
    deg_present = [c for c in ev.columns if c in config.DEGENERATE]
    print(f"degenerate cols present-but-EXCLUDED: {len(deg_present)}")
    print(f"\npositions (round-trips): {len(pos):,}  (tracked entry: {int(pos['tracked'].sum()):,})")
    print("per-wallet entries / exits / positions / median hold / median size$ / net$:")
    for w, g in ev.groupby("wallet"):
        p = pos[pos["wallet"] == w]
        tp = p[p["tracked"]]
        print(f"  {w:10} entries={int(g['is_open'].sum()):>5} exits={int(g['is_close'].sum()):>5} "
              f"pos={len(p):>5} medHold={tp['hold_duration_sec'].median():>6.0f}s "
              f"medSize=${tp['size_usd'].median():>6.1f} net=${p['net_pnl_usd'].sum():>+8.0f}")
    # data hygiene: null rate on the feature set (entries only)
    ent = ev[ev["is_open"]]
    nullrate = ent[feats].isna().mean().sort_values(ascending=False)
    print("\ntop null-rate features at ENTRY (watch for unusable cols):")
    print(nullrate.head(8).round(3).to_dict())
    print("\nC0 OK — event + position matrices built.")


if __name__ == "__main__":
    _self_test()
