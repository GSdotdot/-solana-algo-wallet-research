"""Field-reliability audit over the unified store.

Offline, no API calls. Reads the parquet store (built by store.build_store). The
point is to know, before a field is used as a feature, whether it can be trusted
and how often it actually updates. Per window:
  A. token_snapshots: per-column null rate + zero rate on FULL rows; measured refresh
     cadence = median seconds between observed value CHANGES within a token (on a
     deterministic token sample), + share of rows where the value changed. Zero-rate
     split by `graduated` for the liquidity/reserve/progress fields, which are
     legitimately zero on one side of graduation and suspicious on the other.
  B. wallet_trades: null/zero rates for money/fee columns; per-wallet fee-constancy check
     (tests the assumption that the priority fee is a per-wallet constant).
  C. token_identity: null/empty rates incl. the JSON enrichment blobs.
  D. price_ticks (1s bars): per-token gap distribution + token coverage.
  E. labeled_events: label mix + at-event null rate of key state fields.
Writes field_reliability.json next to the store.

It also quantifies the "degenerate velocity" problem found during the analysis stage:
fields refresh on the REST polling cadence, slower than the snapshot cadence, so the
zero rate of the collector's snapshot-to-snapshot delta columns is reported explicitly.

Usage: python -m store.field_reliability
"""
import glob
import json
import sys
import zlib

import pandas as pd

from common.params import param

from . import STORE_DIR, label_of, wallet_labels

STORE = str(STORE_DIR)
OUT = str(STORE_DIR.parent / "field_reliability.json")
SAMPLE_MOD = param("store.cadence_sample_mod")   # keep 1-in-N tokens (deterministic) for cadence measurement
KEY_SNAP_COLS_SPLIT = ["liquidity", "real_sol_reserves", "virtual_token_reserves",
                       "real_token_reserves", "progress_percent"]
META_COLS = {"snapshot_id", "token_address", "timestamp", "seconds_since_subscription",
             "window", "src_window"}
PRIOR_CHECK_FIELDS = ["price", "volume_1m_usd", "holder", "sniper_count", "bundler_count",
                      "liquidity", "market_cap"]


def snap_parts(window):
    pat = f"{STORE}/snapshots_w1/*.parquet" if window == 1 else f"{STORE}/snapshots_w2/*.parquet"
    return sorted(glob.glob(pat))


def in_sample(tok):
    return zlib.crc32(tok.encode()) % SAMPLE_MOD == 0


def audit_snapshots(window):
    n_rows = 0
    null_c, zero_c, grad_ct = None, None, {c: {} for c in KEY_SNAP_COLS_SPLIT}
    sampled = []
    for p in snap_parts(window):
        df = pd.read_parquet(p)
        num = df.drop(columns=[c for c in df.columns if c in META_COLS], errors="ignore")
        n_rows += len(df)
        nc = num.isna().sum()
        zc = (num == 0).sum()
        null_c = nc if null_c is None else null_c.add(nc, fill_value=0)
        zero_c = zc if zero_c is None else zero_c.add(zc, fill_value=0)
        g = df.get("graduated")
        if g is not None:
            gmask = g.fillna(0).astype(float) > 0
            for c in KEY_SNAP_COLS_SPLIT:
                if c in df:
                    for lab, m in (("grad", gmask), ("curve", ~gmask)):
                        d = grad_ct[c].setdefault(lab, [0, 0])
                        d[0] += int(((df[c] == 0) | df[c].isna())[m].sum())
                        d[1] += int(m.sum())
        keep = df[df.token_address.map(in_sample)]
        if len(keep):
            sampled.append(keep)
    res = {"rows": n_rows,
           "null_rate": (null_c / n_rows).round(4).to_dict(),
           "zero_rate": (zero_c / n_rows).round(4).to_dict(),
           "zero_or_null_by_graduated": {
               c: {lab: round(v[0] / v[1], 4) if v[1] else None for lab, v in d.items()}
               for c, d in grad_ct.items()}}

    # ---- cadence on the token sample ----
    s = pd.concat(sampled, ignore_index=True).sort_values(["token_address", "timestamp"])
    res["cadence_sample"] = {"tokens": int(s.token_address.nunique()), "rows": len(s)}
    same_tok = s.token_address.eq(s.token_address.shift())
    cad = {}
    for c in s.columns:
        if c in META_COLS:
            continue
        v = s[c]
        changed = v.ne(v.shift()) & same_tok & v.notna() & v.shift().notna()
        # seconds since previous change within token: diff of change timestamps per token
        ch = s.loc[changed, ["token_address", "timestamp"]]
        iv = ch.groupby("token_address")["timestamp"].diff().dropna()
        obs = same_tok & v.notna() & v.shift().notna()
        cad[c] = {"median_change_interval_s": float(iv.median()) if len(iv) else None,
                  "p75_s": float(iv.quantile(.75)) if len(iv) else None,
                  "pct_rows_changed": round(float(changed.sum() / max(obs.sum(), 1)), 4),
                  "n_changes": int(changed.sum())}
    res["cadence"] = cad
    return res


def audit_wallet_trades():
    wt = pd.read_parquet(f"{STORE}/wallet_trades.parquet")
    labels = wallet_labels()
    wt["wallet_label"] = wt["wallet_address"].map(lambda a: label_of(a, labels))
    out = {}
    for w, g in wt.groupby("window"):
        money = ["amount_token", "amount_sol", "price_per_token", "volume_usd", "tx_fee_sol",
                 "priority_fee_sol", "prio_bribe_fee_usd", "position_pnl_usd", "net_pnl_usd",
                 "slot_number", "pool_address", "dex_program", "snapshot_ref_id",
                 "has_pre_context", "hold_duration_sec"]
        d = {c: {"null": round(float(g[c].isna().mean()), 4),
                 "zero": round(float((g[c] == 0).mean()), 4) if g[c].dtype.kind in "if" else None}
             for c in money if c in g}
        # fee-constancy: distinct fee values per wallet (assumption under test: the
        # priority fee is a per-wallet constant, not a per-transaction choice)
        d["fee_nunique_per_wallet"] = {
            lab: {"tx_fee": int(x.tx_fee_sol.nunique()), "prio_fee": int(x.priority_fee_sol.nunique()),
                  "trades": len(x)}
            for lab, x in g.groupby("wallet_label")}
        d["direction_values"] = g.direction.value_counts().to_dict()
        out[f"w{w}"] = d
    return out


def audit_identity():
    idc = pd.read_parquet(f"{STORE}/token_identity.parquet")
    idj = pd.read_parquet(f"{STORE}/token_identity_json.parquet")
    out = {}
    for w in (1, 2):
        g = idc[idc.window == w]
        j = idj[idj.window == w]
        d = {c: round(float(g[c].isna().mean()), 4) for c in
             ["creation_time", "creator_wallet", "graduated_time", "pool_address",
              "first_seen_timestamp", "discovered_via_wallet", "platform_id", "decimals"]}
        for c in ["ohlcv_backfill_json", "recent_trades_json", "top_traders_json", "exit_liquidity_json"]:
            populated = j[c].notna() & (j[c].astype(str).str.len() > 2)     # "{}" / "[]" = empty
            d[c + "_populated"] = round(float(populated.mean()), 4)
        d["tokens"] = len(g)
        out[f"w{w}"] = d
    return out


def audit_ticks():
    out = {}
    for w in (1, 2):
        tk = pd.read_parquet(f"{STORE}/ticks_w{w}.parquet",
                             columns=["token_address", "unix_time", "received_at"])
        tk = tk.sort_values(["token_address", "unix_time"])
        gap = tk.groupby("token_address")["unix_time"].diff().dropna()
        out[f"w{w}"] = {"rows": len(tk), "tokens": int(tk.token_address.nunique()),
                        "gap_median_s": float(gap.median()), "gap_p90_s": float(gap.quantile(.9)),
                        "gap_p99_s": float(gap.quantile(.99)),
                        "recv_lag_median_s": float((tk.received_at - tk.unix_time).median())}
    return out


def audit_events():
    ev = pd.read_parquet(f"{STORE}/labeled_events.parquet",
                         columns=["window", "label", "price", "market_cap", "liquidity", "holder",
                                  "volume_1m_usd", "has_pre_context"])
    out = {}
    for w, g in ev.groupby("window"):
        out[f"w{w}"] = {"rows": len(g), "labels": g.label.value_counts().to_dict(),
                        "state_null_rate": {c: round(float(g[c].isna().mean()), 4)
                                            for c in ["price", "market_cap", "liquidity", "holder",
                                                      "volume_1m_usd"]},
                        "has_pre_context_rate": round(float(g.has_pre_context.fillna(0).astype(float).mean()), 4)}
    return out


def main():
    res = {"wallet_trades": audit_wallet_trades(), "token_identity": audit_identity(),
           "price_ticks": audit_ticks(), "labeled_events": audit_events()}
    for w in (1, 2):
        print(f"== auditing snapshots w{w} ...")
        res[f"snapshots_w{w}"] = audit_snapshots(w)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("== written field_reliability.json")

    # console digest: the headline reliability facts
    for w in (1, 2):
        r = res[f"snapshots_w{w}"]
        print(f"\n-- snapshots w{w}: {r['rows']} rows; cadence sample {r['cadence_sample']}")
        print("   zero/null by graduated:", r["zero_or_null_by_graduated"])
        worst = sorted(r["cadence"].items(),
                       key=lambda kv: -(kv[1]["median_change_interval_s"] or 0))[:8]
        print("   slowest cadences:", [(k, v["median_change_interval_s"]) for k, v in worst])
        key = {k: r["cadence"][k]["median_change_interval_s"] for k in PRIOR_CHECK_FIELDS
               if k in r["cadence"]}
        print("   prior-check cadences:", key)
        vel = {k: v for k, v in r["zero_rate"].items() if "velocity" in k or "delta" in k}
        print("   velocity/delta zero rates:", dict(list(vel.items())[:8]))
    print("\n-- 1s bars:", res["price_ticks"])
    print("-- identity:", res["token_identity"])


if __name__ == "__main__":
    sys.exit(main())
