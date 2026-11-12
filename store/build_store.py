"""Unified research store + per-wallet coverage inventory.

Offline, no API calls. Reads:
  - Window 1: the frozen collector database (SQLite, opened read-only)
  - Window 2: the collector's periodic CSV exports (`window_<since>_<until>/` folders:
    sequential incremental windows, plus full dumps)
Writes a parquet store (gitignored):
  wallet_trades / labeled_events / token_identity (+_json) / subscription_state
  snapshots_w1/ snapshots_w2/ (per-part)  ticks_w1 ticks_w2  + manifest.json + coverage.json

Dedupe keys (the full dumps overlap the sequential windows by construction):
  token_snapshots -> snapshot_id ("token:ts"); wallet_trades -> tx_signature;
  labeled_events -> event_id ("txsig:LABEL"); price_ticks -> (token_address, unix_time);
  token_identity/subscription_state -> token_address (keep LAST occurrence = most complete row).
The snapshot seen-set is bounded: only ids with a timestamp at or before the last
full dump can ever collide with one, so later ids are not retained in memory.

Per-field observed-at times are NOT materialised here: refresh cadences are MEASURED
empirically by store.field_reliability and derived there; the working priors are
recorded in manifest["field_cadence_priors"] until then.

Usage: python -m store.build_store            (or --coverage-only)
"""
import glob
import json
import os
import re
import sqlite3
import sys

import pandas as pd

from common.params import param

from . import STORE_DIR, W1_DB, W2_ROOT, label_of, wallet_labels

OUT = str(STORE_DIR)
JSON_COLS = ["ohlcv_backfill_json", "recent_trades_json", "top_traders_json", "exit_liquidity_json"]
SNAPSHOT_CHUNK_ROWS = 250_000     # rows per parquet part (memory bound, not a tuned value)
SEQ_GAP_TOLERANCE_SEC = 2         # sequential windows should abut; larger gaps are reported

# Working priors for each field's refresh cadence (seconds), to be replaced by the
# measured values from store.field_reliability. {field: seconds}; redacted.
CADENCE_PRIORS_SEC = param("analysis.features.cadence_sec")

STR_COL_RE = re.compile(r"(address|signature|_id$|symbol|name|label|^src|direction|tx_type|dex|"
                        r"platform|uri|twitter|website|description|json|payload|grace_tier|source)")


def harmonize_object_cols(df):
    """Window 1 (SQLite) and window 2 (CSV) disagree on some column dtypes (e.g. the
    reserve columns are TEXT in the database). After concat, make every object column
    either fully numeric (only when no non-empty value is lost) or fully string, so
    parquet conversion cannot fail."""
    for c in df.columns:
        if df[c].dtype != object or STR_COL_RE.search(c):
            continue
        s = df[c]
        nonempty = s.notna() & (s.astype(str).str.strip() != "")
        conv = pd.to_numeric(s, errors="coerce")
        if conv[nonempty].notna().all():
            df[c] = conv
        else:
            df[c] = s.where(s.isna(), s.astype(str))
    return df


def read_csv_gz(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:
        if "No columns" in str(e):
            return pd.DataFrame()
        raise


def w2_window_dirs():
    seq, dumps = [], []
    for d in sorted(glob.glob(os.path.join(str(W2_ROOT), "window_*"))):
        m = re.match(r"window_(\d+)_(\d+)$", os.path.basename(d))
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        (dumps if a == 0 else seq).append((a, b, d))
    seq.sort()
    dumps.sort(key=lambda t: t[1])
    return seq, dumps


def main():
    os.makedirs(f"{OUT}/snapshots_w1", exist_ok=True)
    os.makedirs(f"{OUT}/snapshots_w2", exist_ok=True)
    manifest = {"built_from": {"w1": W1_DB.name, "w2": W2_ROOT.name},
                "field_cadence_priors": CADENCE_PRIORS_SEC, "parts": {}, "dedupe": {}}

    # ---------------- window 1 (SQLite, frozen) ----------------
    con = sqlite3.connect(f"file:{W1_DB}?mode=ro", uri=True)
    small_w1 = {}
    for tbl in ("wallet_trades", "labeled_events", "token_identity", "subscription_state"):
        small_w1[tbl] = pd.read_sql(f"select * from {tbl}", con)
        small_w1[tbl]["window"] = 1
    n_parts, w1_snap_rows = 0, 0
    for chunk in pd.read_sql("select * from token_snapshots order by timestamp", con,
                             chunksize=SNAPSHOT_CHUNK_ROWS):
        chunk["window"] = 1
        harmonize_object_cols(chunk)
        chunk.to_parquet(f"{OUT}/snapshots_w1/part-{n_parts:03d}.parquet", compression="zstd", index=False)
        w1_snap_rows += len(chunk)
        n_parts += 1
    ticks_w1 = pd.read_sql("select * from price_ticks", con)
    ticks_w1["window"] = 1
    ticks_w1.to_parquet(f"{OUT}/ticks_w1.parquet", compression="zstd", index=False)
    con.close()
    manifest["parts"]["snapshots_w1"] = {"parts": n_parts, "rows": w1_snap_rows}
    manifest["parts"]["ticks_w1"] = {"rows": len(ticks_w1)}
    print(f"== W1: snapshots {w1_snap_rows} rows/{n_parts} parts; 1s bars {len(ticks_w1)}; "
          f"trades {len(small_w1['wallet_trades'])}; identity {len(small_w1['token_identity'])}")

    # ---------------- window 2 (CSV windows; sequential first, dumps fill gaps) ----------------
    seq, dumps = w2_window_dirs()
    print(f"== W2: {len(seq)} sequential + {len(dumps)} dump dirs")
    gaps = [(p[1], c[0]) for p, c in zip(seq, seq[1:]) if abs(c[0] - p[1]) > SEQ_GAP_TOLERANCE_SEC]
    if gaps:
        print(f"   WARNING sequential gaps: {gaps}")
    manifest["w2_sequential_gaps"] = gaps

    # only snapshot ids at/below the last dump's end can collide with a dump
    DUMP_MAX_TS = max((b for _, b, _ in dumps), default=0) + 10
    seen_snap = set()
    seen = {"wt": set(), "ev": set(), "tick": set()}
    acc = {"wallet_trades": [], "labeled_events": [], "token_identity": [],
           "subscription_state": [], "ticks": []}
    recon_bad, dd_stats = [], {"snapshots_dropped": 0, "wt_dropped": 0, "ev_dropped": 0, "tick_dropped": 0}

    for a, b, d in seq + dumps:
        name = os.path.basename(d)
        summ = json.load(open(os.path.join(d, "summary.json")))
        exp = summ.get("rows_exported", {})
        # snapshots: stream one window at a time, drop ids already seen (dump overlap)
        snap = read_csv_gz(os.path.join(d, "token_snapshots.csv.gz"))
        if exp.get("token_snapshots") is not None and exp["token_snapshots"] != len(snap):
            recon_bad.append((name, "token_snapshots", exp["token_snapshots"], len(snap)))
        if len(snap):
            dup = snap.snapshot_id.isin(seen_snap)
            dd_stats["snapshots_dropped"] += int(dup.sum())
            snap = snap[~dup]
            if len(snap):
                snap = snap.copy()
                snap["window"] = 2
                snap["src_window"] = name
                harmonize_object_cols(snap)
                snap.to_parquet(f"{OUT}/snapshots_w2/{name}.parquet", compression="zstd", index=False)
                seen_snap.update(snap.loc[snap.timestamp <= DUMP_MAX_TS, "snapshot_id"])
                manifest["parts"].setdefault("snapshots_w2", {"parts": []})["parts"].append(
                    {"src": name, "rows": len(snap),
                     "ts_range": [int(snap.timestamp.min()), int(snap.timestamp.max())]})
        # small tables
        for tbl, key, seen_key in (("wallet_trades", "tx_signature", "wt"),
                                   ("labeled_events", "event_id", "ev")):
            df = read_csv_gz(os.path.join(d, f"{tbl}.csv.gz"))
            if exp.get(tbl) is not None and exp[tbl] != len(df):
                recon_bad.append((name, tbl, exp[tbl], len(df)))
            if len(df):
                dup = df[key].isin(seen[seen_key])
                dd_stats[f"{seen_key}_dropped"] += int(dup.sum())
                df = df[~dup].copy()
                seen[seen_key].update(df[key])
                df["src_window"] = name
                acc[tbl].append(df)
        for tbl in ("token_identity", "subscription_state"):
            df = read_csv_gz(os.path.join(d, f"{tbl}.csv.gz"))
            if len(df):
                df["src_window"] = name
                acc[tbl].append(df)
        tk = read_csv_gz(os.path.join(d, "price_ticks.csv.gz"))
        if len(tk):
            key = list(zip(tk.token_address, tk.unix_time))
            dup = pd.Series([k in seen["tick"] for k in key], index=tk.index)
            dd_stats["tick_dropped"] += int(dup.sum())
            tk = tk[~dup]
            seen["tick"].update(k for k, isdup in zip(key, dup) if not isdup)
            acc["ticks"].append(tk)
        print(f"   {name}: snap +{len(snap) if isinstance(snap, pd.DataFrame) else 0}")

    # every export folder carries a summary.json with the row counts it wrote; a
    # mismatch here means a truncated or corrupted transfer
    print(f"== RECONCILE vs summary.json: {'OK' if not recon_bad else recon_bad}")
    print(f"== DEDUPE: {dd_stats}")
    manifest["reconcile_mismatches"] = recon_bad
    manifest["dedupe"] = dd_stats

    # window-2 small tables -> parquet (identity/subscription: keep LAST row per token)
    wt2 = pd.concat(acc["wallet_trades"], ignore_index=True)
    ev2 = pd.concat(acc["labeled_events"], ignore_index=True)
    id2 = (pd.concat(acc["token_identity"], ignore_index=True)
           .drop_duplicates("token_address", keep="last"))
    ss2 = (pd.concat(acc["subscription_state"], ignore_index=True)
           .drop_duplicates("token_address", keep="last"))
    tk2 = pd.concat(acc["ticks"], ignore_index=True)
    for df in (wt2, ev2, id2, ss2, tk2):
        df["window"] = 2

    wt = harmonize_object_cols(pd.concat([small_w1["wallet_trades"], wt2], ignore_index=True))
    ev = harmonize_object_cols(pd.concat([small_w1["labeled_events"], ev2], ignore_index=True))
    ident = harmonize_object_cols(pd.concat([small_w1["token_identity"], id2], ignore_index=True))
    ss = harmonize_object_cols(pd.concat([small_w1["subscription_state"], ss2], ignore_index=True))
    wt.to_parquet(f"{OUT}/wallet_trades.parquet", compression="zstd", index=False)
    ev.to_parquet(f"{OUT}/labeled_events.parquet", compression="zstd", index=False)
    ident.drop(columns=JSON_COLS).to_parquet(f"{OUT}/token_identity.parquet", compression="zstd", index=False)
    ident[["token_address", "window"] + JSON_COLS].to_parquet(
        f"{OUT}/token_identity_json.parquet", compression="zstd", index=False)
    ss.to_parquet(f"{OUT}/subscription_state.parquet", compression="zstd", index=False)
    tk2.to_parquet(f"{OUT}/ticks_w2.parquet", compression="zstd", index=False)
    manifest["parts"]["wallet_trades"] = {"rows": len(wt), "w1": len(small_w1["wallet_trades"]), "w2": len(wt2)}
    manifest["parts"]["labeled_events"] = {"rows": len(ev)}
    manifest["parts"]["token_identity"] = {"rows": len(ident)}
    manifest["parts"]["subscription_state"] = {"rows": len(ss)}
    manifest["parts"]["ticks_w2"] = {"rows": len(tk2)}

    # global snapshot_id uniqueness assert across all written w2 parts
    import pyarrow.parquet as pq
    ids = pd.concat([pq.read_table(p, columns=["snapshot_id"]).to_pandas()["snapshot_id"]
                     for p in sorted(glob.glob(f"{OUT}/snapshots_w2/*.parquet"))], ignore_index=True)
    n_dup = int(ids.duplicated().sum())
    assert n_dup == 0, f"{n_dup} duplicate snapshot_ids remain in snapshots_w2"
    print(f"== snapshots_w2 global uniqueness: OK ({len(ids)} ids)")
    manifest["parts"]["snapshots_w2"]["rows_total"] = int(len(ids))

    coverage(wt, manifest)


def coverage(wt, manifest):
    """Per wallet x window: how much data there is to decode from (trades, tokens,
    span, trades per day, share of trades that had pre-trade context)."""
    labels = wallet_labels()
    wt["wallet_label"] = wt["wallet_address"].map(lambda a: label_of(a, labels))
    wt["day"] = pd.to_datetime(wt["timestamp"], unit="s", utc=True).dt.date.astype(str)
    cov = {}
    for (lab, win), g in wt.groupby(["wallet_label", "window"]):
        span_days = max((g.timestamp.max() - g.timestamp.min()) / 86400, 1e-9)
        cov.setdefault(lab, {})[f"w{win}"] = {
            "trades": len(g), "buys": int((g.direction.str.upper() == "BUY").sum()),
            "sells": int((g.direction.str.upper() == "SELL").sum()),
            "unique_tokens": int(g.token_address.nunique()),
            "first": pd.to_datetime(g.timestamp.min(), unit="s", utc=True).isoformat(),
            "last": pd.to_datetime(g.timestamp.max(), unit="s", utc=True).isoformat(),
            "span_days": round(span_days, 3),
            "trades_per_day": round(len(g) / span_days, 1),
            "entries": int(g.is_entry.fillna(0).astype(int).sum()) if "is_entry" in g else None,
            "has_pre_context_rate": round(float(g.has_pre_context.fillna(0).astype(int).mean()), 3),
            "per_day": g.groupby("day").size().to_dict(),
        }
    with open(f"{OUT}/coverage.json", "w") as f:
        json.dump(cov, f, indent=1, default=str)
    if manifest is not None:
        with open(f"{OUT}/manifest.json", "w") as f:
            json.dump(manifest, f, indent=1, default=str)

    print("\n== COVERAGE (trades per wallet x window) ==")
    rows = []
    for lab in sorted(cov):
        r = {"wallet": lab}
        for w in ("w1", "w2"):
            c = cov[lab].get(w)
            r[f"{w}_trades"] = c["trades"] if c else 0
            r[f"{w}_tokens"] = c["unique_tokens"] if c else 0
            r[f"{w}_per_day"] = c["trades_per_day"] if c else 0.0
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n== STORE written; manifest.json + coverage.json alongside")


if __name__ == "__main__":
    if "--coverage-only" in sys.argv:
        coverage(pd.read_parquet(f"{OUT}/wallet_trades.parquet"), None)
    else:
        sys.exit(main())
