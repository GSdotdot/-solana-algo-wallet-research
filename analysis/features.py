"""Central feature factory: the FULL data inventory, built once, consumed by every module.

Analysis doctrine: modules must start from the COMPLETE data universe and re-test it,
never a curated subset. This module is that universe, in causal, analysis-ready form.
C3 (`exit.py`) is the methodological reference; this generalises its on-chain
engineering to the WHOLE catalog and adds the three REST enrichment blobs
(pre-discovery price history, top traders, recent trades).

What it produces:
  snapshot_features(frame, times)  : causal on-chain vector at arbitrary times: ALL levels +
                                     cadence-matched deltas + the full order-flow ladder
                                     (counts/volumes/sell-fraction/net-flow/imbalance per tf).
  token_enrichment(identity)       : per-token static features from the REST JSON blobs
                                     (pre-discovery history, top-trader field incl. the tracked
                                     wallets' own standing, recent-trade microstructure).
  entry_state_full(events)         : every ENTRY/FLIP with the FULL on-chain vector + enrichment.

"Causal" throughout means: a feature at time t uses only snapshots at or before t.

Run `python -m analysis.features` for a coverage self-test.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common.params import param

from . import config, data

EPS = 1e-9
P = config.P


def _blob_fields(group: str) -> dict:
    """Provider JSON keys for one enrichment blob (redacted in the public copy)."""
    return param(f"data_provider.response_fields.{group}")


# --- snapshot loading (full catalog) ---------------------------------------------
def load_snapshot_frames() -> dict[str, pd.DataFrame]:
    """Every catalog column from token_snapshots, grouped by token, time-sorted, numeric."""
    cols = config.all_snapshot_cols()
    con = data._conn()
    df = pd.read_sql(f"SELECT {','.join(cols)} FROM token_snapshots ORDER BY token_address, timestamp", con)
    con.close()
    for c in df.columns:
        if c != "token_address":
            df[c] = pd.to_numeric(df[c], errors="coerce")            # reserves are TEXT
    return {t: g.reset_index(drop=True) for t, g in df.groupby("token_address", sort=False)}


def feature_names() -> list[str]:
    """The full feature catalog produced by snapshot_features + token_enrichment — the names
    every module can import as its starting universe (used e.g. by entry.model_features())."""
    names = [c for c in config.all_snapshot_cols() if c not in ("token_address", "timestamp")]
    names += config.delta_feature_names()
    for tf in config.TIMEFRAMES:
        names += [f"sell_frac_{tf}", f"net_flow_{tf}", f"log_vol_{tf}",
                  f"trade_imbalance_{tf}", f"n_trades_{tf}"]
    names += ["vol_accel_1m", "bf_n_candles", "bf_pre_return", "bf_pre_vol_usd",
              "bf_active_frac", "bf_volatility", "tt_n", "tt_pnl_sum", "tt_frac_profitable",
              "tt_top1_vol_share", "tt_n_target_wallets", "rt_n", "rt_sell_frac",
              "rt_unique_owners", "rt_med_usd"]
    return names


def feature_groups() -> dict[str, list[str]]:
    """Feature families used to attribute decodability (C2), exit-timing lift (C3) and
    sizing variance (C4) to a KIND of data rather than to individual columns."""
    class_fields = [f"{c}_held_percentage" for c in config.TRADER_CLASSES]
    return {
        "structural": ["price", "market_cap", "fdv", "liquidity", "holder",
                       "top10_holder_percentage", "progress_percent", "graduated"],
        "supply_reserve": config.CAT_SUPPLY_RESERVE,
        "classification": config.CAT_CLASS_LEVELS + config.delta_feature_names(class_fields),
        "orderflow": (config.orderflow_cols() + ["buy_sell_ratio_1m", "buy_sell_ratio_5m"]
                      + [f"{p}{tf}" for tf in config.TIMEFRAMES
                         for p in ("sell_frac_", "net_flow_", "log_vol_", "trade_imbalance_", "n_trades_")]
                      + ["vol_accel_1m"]),
        "enrichment": ["bf_n_candles", "bf_pre_return", "bf_pre_vol_usd", "bf_active_frac",
                       "bf_volatility", "tt_n", "tt_pnl_sum", "tt_frac_profitable",
                       "tt_top1_vol_share", "tt_n_target_wallets", "rt_n", "rt_sell_frac",
                       "rt_unique_owners", "rt_med_usd"],
    }


def _ff(sts: np.ndarray, times: np.ndarray, lb: int = 0) -> np.ndarray:
    """Indices of the snapshot at-or-before (time - lb). Causal forward-fill."""
    return np.clip(np.searchsorted(sts, times - lb, side="right") - 1, 0, len(sts) - 1)


def snapshot_features(frame: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    """Causal full on-chain feature vector at each query time. ALL absolute levels +
    cadence-matched deltas (windows >= each field's refresh, so they are real movements) +
    the complete order-flow ladder (derived sell-fraction / net-flow / imbalance / log-vol /
    acceleration at every timeframe)."""
    times = np.asarray(times, float)
    sts = frame["timestamp"].to_numpy()
    arr = {c: frame[c].to_numpy() for c in frame.columns if c not in ("token_address", "timestamp")}
    idx = _ff(sts, times)
    out = {c: v[idx] for c, v in arr.items()}                        # absolute levels (all)

    for f, wins in config.DELTA_FIELDS.items():                     # cadence-matched deltas
        if f not in arr:
            continue
        now = arr[f][idx]
        pct = f in config.PCT_DELTA_FIELDS
        for w in wins:
            prev = arr[f][_ff(sts, times, w)]
            if pct:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[f"{f}_d{w}_pct"] = np.where(prev > 0, now / prev - 1.0, np.nan)
            else:
                out[f"{f}_d{w}"] = now - prev

    for tf in config.TIMEFRAMES:                                    # order-flow ladder, derived
        vb, vs = arr.get(f"volume_buy_{tf}_usd"), arr.get(f"volume_sell_{tf}_usd")
        vt, nb, ns = arr.get(f"volume_{tf}_usd"), arr.get(f"buy_{tf}"), arr.get(f"sell_{tf}")
        if vb is not None and vs is not None:
            b, s = vb[idx], vs[idx]
            out[f"sell_frac_{tf}"] = s / (b + s + EPS)
            out[f"net_flow_{tf}"] = b - s
        if vt is not None:
            out[f"log_vol_{tf}"] = np.log1p(np.clip(vt[idx], 0, None))
        if nb is not None and ns is not None:
            b, s = nb[idx], ns[idx]
            out[f"trade_imbalance_{tf}"] = (b - s) / (b + s + EPS)
            out[f"n_trades_{tf}"] = b + s
    if "volume_1m_usd" in arr:                                      # 1m volume acceleration
        out["vol_accel_1m"] = (arr["volume_1m_usd"][idx]
                               - arr["volume_1m_usd"][_ff(sts, times, config.VOL_ACCEL_LAG_SEC)])
    return pd.DataFrame(out)


# --- REST enrichment parsers (one-time per-token snapshots) -----------------------
# Each parser maps the provider's JSON keys (config, redacted) onto neutral column
# names, so nothing downstream depends on the provider's vocabulary.
def _blob_items(js: str) -> list:
    if not js:
        return []
    env = param("data_provider.response_fields.envelope")
    items_key = param("data_provider.response_fields.blob_items")
    return ((json.loads(js) or {}).get(env) or {}).get(items_key, [])


def parse_backfill(js: str) -> pd.DataFrame:
    """Pre-discovery price history -> columns o, h, l, c, v, t (bar open time)."""
    F = _blob_fields("ohlcv_backfill")
    df = pd.DataFrame(_blob_items(js))
    return df.rename(columns={F["open"]: "o", F["high"]: "h", F["low"]: "l",
                              F["close"]: "c", F["volume"]: "v", F["bar_time"]: "t"})


def parse_top_traders(js: str) -> pd.DataFrame:
    """Top traders on the token -> columns owner, realized_pnl, volume (+ the rest)."""
    F = _blob_fields("top_traders")
    df = pd.DataFrame(_blob_items(js))
    return df.rename(columns={F["owner"]: "owner", F["realized_pnl"]: "realized_pnl",
                              F["volume"]: "volume"})


def parse_recent_trades(js: str) -> pd.DataFrame:
    """Recent swaps on the token -> one neutral row per trade."""
    F = _blob_fields("recent_trades")
    arr = json.loads(js) if js else []
    rows = [{"time": t.get(F["block_time"]), "side": t.get(F["side"]), "owner": t.get(F["owner"]),
             "source": t.get(F["source"]),
             "base_amt": (t.get(F["base_leg"]) or {}).get(F["leg_ui_amount"]),
             "usd": abs((t.get(F["quote_leg"]) or {}).get(F["leg_ui_amount"]) or 0) * ((t.get(F["quote_price"])) or 0),
             "price": t.get(F["token_price"]) or t.get(F["base_price"])} for t in arr]
    return pd.DataFrame(rows)


def token_enrichment(identity: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-token static features from the REST JSON blobs (causality caveat: captured ~discovery
    time — use as token CONTEXT / pre-entry, not as a live in-hold signal)."""
    if identity is None:
        con = data._conn()
        identity = pd.read_sql("SELECT token_address, first_seen_timestamp, ohlcv_backfill_json, "
                               "top_traders_json, recent_trades_json FROM token_identity", con)
        con.close()
    targets = set(config.WALLET_LABELS)
    rows = []
    for r in identity.to_dict("records"):
        tok = r["token_address"]; rec = {"token_address": tok}
        bf = parse_backfill(r.get("ohlcv_backfill_json"))
        if len(bf) >= 2 and "c" in bf:
            c = pd.to_numeric(bf["c"], errors="coerce").to_numpy()
            v = pd.to_numeric(bf.get("v", 0), errors="coerce").to_numpy()
            ret = c[c > 0]
            rec["bf_n_candles"] = len(bf)
            rec["bf_pre_return"] = (c[-1] / c[0] - 1) if c[0] > 0 else np.nan
            rec["bf_pre_vol_usd"] = float(np.nansum(v))
            rec["bf_active_frac"] = float(np.mean(v > 0))            # how much of history traded
            rec["bf_volatility"] = float(np.nanstd(np.diff(np.log(ret)))) if len(ret) > 3 else np.nan
        tt = parse_top_traders(r.get("top_traders_json"))
        if len(tt):
            pnl = pd.to_numeric(tt.get("realized_pnl", np.nan), errors="coerce")
            vol = pd.to_numeric(tt.get("volume", 0), errors="coerce")
            rec["tt_n"] = len(tt)
            rec["tt_pnl_sum"] = float(pnl.sum())
            rec["tt_frac_profitable"] = float((pnl > 0).mean())
            rec["tt_top1_vol_share"] = float(vol.max() / (vol.sum() + EPS))
            owners = set(tt.get("owner", pd.Series(dtype=str)))
            rec["tt_n_target_wallets"] = len(owners & targets)       # tracked wallets among top traders
        rt = parse_recent_trades(r.get("recent_trades_json"))
        if len(rt):
            side = rt["side"].astype(str)
            rec["rt_n"] = len(rt)
            rec["rt_sell_frac"] = float((side == _blob_fields("recent_trades")["sell_side_value"]).mean())
            rec["rt_unique_owners"] = int(rt["owner"].nunique())
            rec["rt_med_usd"] = float(pd.to_numeric(rt["usd"], errors="coerce").median())
        rows.append(rec)
    return pd.DataFrame(rows)


# --- entry-state matrix with the FULL inventory ----------------------------------
def entry_state_full(events: pd.DataFrame | None = None, frames: dict | None = None,
                     tol: int | None = None) -> pd.DataFrame:
    """Every ENTRY/FLIP with the FULL causal on-chain vector (snapshot at-or-before entry; for
    first-touch entries the nearest snapshot is the at-trade poll ~entry, flagged) + per-token
    enrichment. Replaces C0's small state recovery with the complete catalog. `tol` is how far
    (seconds) the nearest snapshot may sit from the entry for the row to count as covered."""
    if tol is None:
        tol = P("features.entry_coverage_tol_sec")       # tuned privately
    if events is None:
        events = data.load_events()
    if frames is None:
        frames = load_snapshot_frames()
    ent = events[events["label"].isin(("ENTRY", "FLIP"))].copy().reset_index(drop=True)
    feat_parts, covered = [], []
    for r in ent.itertuples():
        fr = frames.get(r.token_address)
        if fr is None or not len(fr):
            feat_parts.append(None); covered.append(False); continue
        f = snapshot_features(fr, np.array([r.timestamp]))
        nearest = int(np.min(np.abs(fr["timestamp"].to_numpy() - r.timestamp)))
        covered.append(nearest <= max(tol, 0))
        feat_parts.append(f.iloc[0])
    feats = pd.DataFrame([p if p is not None else {} for p in feat_parts]).reset_index(drop=True)
    ent = pd.concat([ent[["wallet", "token_address", "timestamp", "label", "volume_usd"]],
                     feats], axis=1)
    ent["covered"] = covered
    enr = token_enrichment()
    return ent.merge(enr, on="token_address", how="left")


def _self_test() -> None:
    print("loading full snapshot frames ...")
    frames = load_snapshot_frames()
    ncols = len(next(iter(frames.values())).columns)
    print(f"  tokens={len(frames):,}  raw catalog cols/token={ncols}")

    ev = data.load_events()
    ent = entry_state_full(ev, frames)
    feat_cols = [c for c in ent.columns if c not in
                 ("wallet", "token_address", "timestamp", "label", "volume_usd", "covered")]
    print(f"\nentry_state_full: {len(ent):,} entries x {len(feat_cols)} features "
          f"(coverage {ent['covered'].mean():.0%})")
    print("  feature families present:")
    for fam, key in [("levels", lambda c: c in config.all_snapshot_cols()),
                     ("deltas", lambda c: "_d" in c and c[-1].isdigit() or c.endswith("_pct")),
                     ("orderflow-derived", lambda c: c.startswith(("sell_frac_", "net_flow_", "log_vol_", "trade_imbalance_", "n_trades_"))),
                     ("enrichment(JSON)", lambda c: c.startswith(("bf_", "tt_", "rt_")))]:
        cs = [c for c in feat_cols if key(c)]
        print(f"    {fam:20} ({len(cs)}): {cs[:6]}{' ...' if len(cs) > 6 else ''}")

    enr = token_enrichment()
    print(f"\ntoken_enrichment: {len(enr):,} tokens; non-null coverage:")
    for c in [c for c in enr.columns if c != "token_address"]:
        print(f"    {c:22} {enr[c].notna().mean()*100:>4.0f}%  median={pd.to_numeric(enr[c],errors='coerce').median():.4g}")
    tgt = enr["tt_n_target_wallets"].fillna(0)
    print(f"\n  tokens where >=1 tracked wallet is a top-trader: {(tgt>=1).sum()} "
          f"({(tgt>=1).mean()*100:.0f}%) - direct C5 cross-wallet signal")
    print("\nfeatures OK - full inventory built.")


if __name__ == "__main__":
    _self_test()
