"""Stage 1 — periodic export pipeline.

Standalone and scheduler-friendly: run on a fixed interval as a SEPARATE process
from the collector. WAL mode lets this read concurrently with the live writer.
Dumps each table's rows for the window [since, until) to gzipped CSV in a
timestamped folder, writes a summary JSON, advances a watermark so each run exports
only new rows, and (optionally) copies the folder off the collection machine.

Usage:
  python -m collector.export                 # incremental: since=watermark, until=now
  python -m collector.export --full          # everything (end-of-run snapshot; no watermark change)
  python -m collector.export --out DIR --scp TARGET

The core tables (wallet_trades, token_snapshots, labeled_events, token_identity)
are never pruned by the collector, so incremental export loses nothing. price_ticks
IS rolling-window pruned (the targeted-B price-bar archive, TICK_RETENTION_HRS), but
retention is much longer than the export interval, so each window exports well
before it ages out.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
import subprocess
import time
from pathlib import Path

from . import config

# table -> timestamp column used to window the export (None = dump whole table)
TIME_COLUMN = {
    "wallet_trades": "timestamp",
    "token_snapshots": "timestamp",
    "labeled_events": "timestamp",
    "price_ticks": "unix_time",
    "token_identity": "first_seen_timestamp",
    "subscription_state": None,        # small current-state table; dump whole
}

STATE_FILE = config.DATA_DIR / "export_state.json"


def _read_watermark() -> int:
    if STATE_FILE.exists():
        try:
            return int(json.loads(STATE_FILE.read_text())["last_until"])
        except (ValueError, KeyError, OSError):
            pass
    return 0


def _write_watermark(until: int) -> None:
    STATE_FILE.write_text(json.dumps({"last_until": until}))


def _dump_table(conn: sqlite3.Connection, table: str, time_col: str | None,
                since: int, until: int, out_path: Path) -> int:
    if time_col is None:
        cur = conn.execute(f"SELECT * FROM {table}")
    else:
        cur = conn.execute(
            f"SELECT * FROM {table} WHERE {time_col} >= ? AND {time_col} < ? ORDER BY {time_col}",
            (since, until))
    cols = [d[0] for d in cur.description]
    n = 0
    with gzip.open(out_path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row in cur:
            w.writerow(row)
            n += 1
    return n


def export(db_path: Path, out_root: Path, since: int, until: int,
           scp_target: str | None = None, advance_watermark: bool = True) -> dict:
    folder = out_root / f"window_{since}_{until}"
    folder.mkdir(parents=True, exist_ok=True)

    # read-only WAL connection (safe alongside the live collector)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=1;")
        counts = {
            table: _dump_table(conn, table, tcol, since, until, folder / f"{table}.csv.gz")
            for table, tcol in TIME_COLUMN.items()
        }
    finally:
        conn.close()

    summary = {
        "generated_at": int(time.time()),
        "window": {"since": since, "until": until,
                   "since_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(since)),
                   "until_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(until))},
        "rows_exported": counts,
        "db_size_bytes": db_path.stat().st_size if db_path.exists() else None,
        "full_dump": since == 0 and not advance_watermark,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))

    if scp_target:
        try:
            subprocess.run(["scp", "-r", str(folder), scp_target], check=True, timeout=600)
            summary["scp"] = f"ok -> {scp_target}"
        except (subprocess.SubprocessError, OSError) as e:
            summary["scp"] = f"FAILED: {e!r}"

    if advance_watermark:
        _write_watermark(until)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Periodic export")
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--out", default=str(config.DATA_DIR / "exports"))
    ap.add_argument("--since", type=int, default=None, help="window start (unix); default=watermark")
    ap.add_argument("--until", type=int, default=None, help="window end (unix); default=now")
    ap.add_argument("--full", action="store_true", help="export everything; don't move watermark")
    ap.add_argument("--scp", default=None, help="optional scp target for the export folder (or set WRE_EXPORT_SCP_TARGET)")
    args = ap.parse_args()

    until = args.until if args.until is not None else int(time.time())
    if args.full:
        since, advance = 0, False
    else:
        since = args.since if args.since is not None else _read_watermark()
        advance = True

    import os
    scp = args.scp or os.environ.get("WRE_EXPORT_SCP_TARGET")
    summary = export(Path(args.db), Path(args.out), since, until, scp, advance)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
