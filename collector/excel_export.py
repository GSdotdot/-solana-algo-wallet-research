"""Excel workbook deliverable: one-shot, run at the end of a collection run.

Reads the SQLite DB (read-only, WAL-safe) and writes a multi-sheet .xlsx:
  Summary             — headline counts + window
  wallet_profiles     — computed per-wallet aggregates (entries/exits/pnl/hold/flip…)
  cross_wallet_overlap— tokens traded by >1 wallet, with entry order
  labeled_events      — the labeled feature matrix (the ML input)
  wallet_trades       — every trade
  token_identity      — tokens (JSON enrichment blobs dropped for readability)
  token_snapshots_1m  — snapshots downsampled to ~1/min, capped (full set is CSV)

price_ticks and the raw JSON blobs stay in the gzip CSV export (collector.export);
Excel is the human-readable summary, not the bulk archive. Uses openpyxl write_only
mode so wide/long sheets stay memory-light.

Usage: python -m collector.excel_export [--db PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

from openpyxl import Workbook

from common.params import param

from . import config
from .util import get_logger

log = get_logger("xlsx")

SNAPSHOT_ROW_CAP = param("collector.export.snapshot_row_cap")   # stay well under Excel's per-sheet row limit
FAST_EXIT_SEC = param("collector.export.fast_exit_sec")         # an exit faster than this counts as a "fast exit"

WALLET_PROFILES_SQL = f"""
SELECT wallet_address,
       COUNT(*)                                   AS trades,
       SUM(is_entry)                              AS entries,
       SUM(is_exit)                               AS exits,
       SUM(is_add)                                AS adds,
       SUM(is_partial_exit)                       AS partial_exits,
       SUM(is_flip)                               AS flips,
       COUNT(DISTINCT token_address)              AS distinct_tokens,
       ROUND(AVG(CASE WHEN is_exit=1 THEN hold_duration_sec END), 1) AS avg_hold_sec_on_exit,
       ROUND(SUM(COALESCE(position_pnl_sol, 0)), 4) AS total_pnl_sol,
       ROUND(100.0 * SUM(CASE WHEN is_exit=1 AND position_pnl_sol>0 THEN 1 ELSE 0 END)
             / NULLIF(SUM(is_exit), 0), 1)         AS win_rate_pct,
       ROUND(100.0 * SUM(is_flip)
             / NULLIF(SUM(is_entry) + SUM(is_flip), 0), 1) AS flip_rate_pct,
       SUM(CASE WHEN is_exit=1 AND hold_duration_sec < {FAST_EXIT_SEC} THEN 1 ELSE 0 END) AS fast_exits
FROM wallet_trades
GROUP BY wallet_address
ORDER BY trades DESC
"""

CROSS_WALLET_SQL = """
WITH first_trade AS (
    SELECT token_address, wallet_address, MIN(timestamp) AS first_ts, COUNT(*) AS trades
    FROM wallet_trades GROUP BY token_address, wallet_address
), overlap AS (
    SELECT token_address FROM first_trade GROUP BY token_address HAVING COUNT(*) > 1
)
SELECT ft.token_address, ft.wallet_address, ft.first_ts, ft.trades,
       RANK() OVER (PARTITION BY ft.token_address ORDER BY ft.first_ts) AS entry_rank
FROM first_trade ft JOIN overlap o ON ft.token_address = o.token_address
ORDER BY ft.token_address, ft.first_ts
"""

IDENTITY_SQL = """
SELECT token_address, token_name, token_symbol, decimals, source, pool_address,
       creator_wallet, creation_time, graduated_time, discovered_via_wallet,
       first_seen_timestamp
FROM token_identity
"""

SNAPSHOTS_1M_SQL = f"""
SELECT * FROM token_snapshots WHERE rowid IN (
    SELECT MIN(rowid) FROM token_snapshots GROUP BY token_address, timestamp/60
) ORDER BY token_address, timestamp LIMIT {SNAPSHOT_ROW_CAP}
"""


def _sheet_from_query(wb: Workbook, title: str, conn: sqlite3.Connection, sql: str) -> int:
    """Stream a query into a new sheet. Returns row count (excl. header)."""
    ws = wb.create_sheet(title=title[:31])  # Excel caps sheet names at 31 chars
    cur = conn.execute(sql)
    ws.append([d[0] for d in cur.description])
    n = 0
    for row in cur:
        ws.append(list(row))
        n += 1
    return n


def export(db_path: Path, out_path: Path) -> dict:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1;")
    wb = Workbook(write_only=True)
    counts: dict[str, int] = {}
    try:
        summary = wb.create_sheet(title="Summary")
        for t in ("wallet_trades", "token_snapshots", "token_identity",
                  "labeled_events", "price_ticks"):
            counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]

        def scalar(q):
            r = conn.execute(q).fetchone()
            return r[0] if r else None

        rng = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM wallet_trades").fetchone()
        summary.append(["Wallet research: run summary"])
        summary.append(["generated", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())])
        summary.append(["wallets (distinct in trades)", scalar("SELECT COUNT(DISTINCT wallet_address) FROM wallet_trades")])
        summary.append(["tokens discovered", counts["token_identity"]])
        summary.append(["wallet_trades", counts["wallet_trades"]])
        summary.append(["labeled_events", counts["labeled_events"]])
        summary.append(["  entries", scalar("SELECT COUNT(*) FROM labeled_events WHERE label='ENTRY'")])
        summary.append(["  exits", scalar("SELECT COUNT(*) FROM labeled_events WHERE label='EXIT'")])
        summary.append(["  flips", scalar("SELECT COUNT(*) FROM labeled_events WHERE label='FLIP'")])
        summary.append(["token_snapshots", counts["token_snapshots"]])
        summary.append(["price_ticks (1s bars)", counts["price_ticks"]])
        if rng and rng[0]:
            summary.append(["first trade (UTC)", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(rng[0]))])
            summary.append(["last trade (UTC)", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(rng[1]))])

        # analytical + data sheets (each isolated so one failure can't void the file)
        for title, sql in (
            ("wallet_profiles", WALLET_PROFILES_SQL),
            ("cross_wallet_overlap", CROSS_WALLET_SQL),
            ("labeled_events", "SELECT * FROM labeled_events ORDER BY timestamp"),
            ("wallet_trades", "SELECT * FROM wallet_trades ORDER BY timestamp"),
            ("token_identity", IDENTITY_SQL),
            ("token_snapshots_1m", SNAPSHOTS_1M_SQL),
        ):
            try:
                counts[f"sheet:{title}"] = _sheet_from_query(wb, title, conn, sql)
            except Exception:  # noqa: BLE001
                log.exception("sheet %s failed", title)

        if counts.get("sheet:token_snapshots_1m", 0) >= SNAPSHOT_ROW_CAP:
            summary.append(["NOTE", f"token_snapshots_1m truncated at {SNAPSHOT_ROW_CAP}; full set in CSV export"])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(out_path)
    finally:
        conn.close()
    log.info("wrote %s (%s)", out_path, {k: v for k, v in counts.items() if k.startswith("sheet:")})
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Excel workbook deliverable")
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--out", default=str(config.DATA_DIR / "exports" /
                                         f"wallet_research_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.xlsx"))
    args = ap.parse_args()
    counts = export(Path(args.db), Path(args.out))
    print(f"Saved {args.out}")
    for k, v in counts.items():
        if k.startswith("sheet:"):
            print(f"  {k[6:]}: {v} rows")


if __name__ == "__main__":
    main()
