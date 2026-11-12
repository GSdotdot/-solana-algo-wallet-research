"""One-shot historical PnL backfill: recompute position_pnl_usd / net_pnl_usd /
prio_bribe_fee_usd / position_pnl_sol for every existing wallet_trades row.

WHY: older rows were written by a SOL-only engine, which (a) booked USDC-quoted
trades' amounts into the SOL columns (inflating SOL PnL for any USDC-quoted trading by
the SOL price), and (b) used full sale proceeds without
pro-rating to the tracked size. This replays every swap through the SAME
PositionBook the live collector now uses (re-parsing raw_payload so the corrected
quote/amount_sol are applied) and writes the USD-canonical results. It is the
historical twin of the live path; one source of truth (positions.py).

SAFE alongside the running collector: WAL + a busy_timeout let this writer and the
collector's inserts serialize. Re-runnable + idempotent (same input -> same output).

Usage:
  python -m collector.recompute_pnl              # DRY RUN: replay + per-wallet report, NO writes
  python -m collector.recompute_pnl --apply      # also UPDATE the rows
  python -m collector.recompute_pnl --db PATH --apply
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict

from common.params import param

from . import config
from .positions import PositionBook
from .wallet_txs import parse_wallet_tx


BUSY_TIMEOUT_MS = param("collector.recompute.busy_timeout_ms")


def recompute(db_path: str, apply: bool = False) -> dict:
    fees = config.load_wallet_fees()
    dex = config.load_dex_programs()
    wl = {w["address"]: w.get("label") or f"wallet_{i}" for i, w in enumerate(config.load_wallets(), 1)}

    conn = sqlite3.connect(db_path)
    conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_MS)}")   # wait out the live writer rather than erroring
    # Replay STRICTLY per (wallet, token) in time order — the only ordering position
    # tracking depends on. Re-parse raw_payload so the corrected quote/amount_sol apply.
    rows = conn.execute(
        "SELECT trade_id, raw_payload FROM wallet_trades "
        "WHERE tx_type='swap' AND direction IN ('BUY','SELL') AND token_address IS NOT NULL "
        "ORDER BY wallet_address, token_address, timestamp, trade_id").fetchall()

    pb = PositionBook()
    updates: list[tuple] = []
    agg = defaultdict(lambda: {"gross": 0.0, "net": 0.0, "fees": 0.0, "sells": 0, "sol_pnl": 0.0, "sol_n": 0})
    parse_fail = 0
    for tid, raw in rows:
        if not raw:
            continue
        try:
            data = json.loads(raw)
            r = parse_wallet_tx(data, dex)
        except Exception:  # noqa: BLE001 - a corrupt payload must not abort the backfill
            parse_fail += 1
            continue
        if r["direction"] not in ("BUY", "SELL") or not r.get("token_address"):
            continue
        w, ts = r.get("wallet_address"), r.get("timestamp")
        if not w or ts is None:
            continue
        fee = config.leg_fee_usd(fees, w, r["direction"])
        d = pb.on_trade(w, r["token_address"], r["direction"],
                        r.get("amount_token") or 0.0, r.get("amount_sol"),
                        r.get("volume_usd") or 0.0, ts,
                        quote=r.get("quote") or "SOL", leg_fee_usd=fee)
        updates.append((d["position_pnl_usd"], d["prio_bribe_fee_usd"],
                        d["net_pnl_usd"], d["position_pnl_sol"], tid))
        a = agg[w]
        if d["net_pnl_usd"] is not None:        # a realized (tracked) sell
            a["gross"] += d["position_pnl_usd"]; a["net"] += d["net_pnl_usd"]; a["sells"] += 1
        if d["prio_bribe_fee_usd"]:
            a["fees"] += d["prio_bribe_fee_usd"]
        if d["position_pnl_sol"] is not None:
            a["sol_pnl"] += d["position_pnl_sol"]; a["sol_n"] += 1

    print(f"replayed {len(updates)} swap legs ({parse_fail} unparseable, skipped)")
    print(f"  {'wallet':8} {'gross_usd':>10} {'fees_usd':>9} {'net_usd':>10} {'sells':>6} {'sol_pnl':>9} {'sol_legs':>8}")
    tg = tn = tf = 0.0
    for w, a in sorted(agg.items(), key=lambda kv: -kv[1]["net"]):
        tg += a["gross"]; tn += a["net"]; tf += a["fees"]
        sol = f"{a['sol_pnl']:+.1f}" if a["sol_n"] else "  (USDC)"
        print(f"  {wl.get(w, 'unlisted'):8} {a['gross']:>+10.1f} {a['fees']:>9.2f} {a['net']:>+10.1f} "
              f"{a['sells']:>6} {sol:>9} {a['sol_n']:>8}")
    print(f"  {'TOTAL':8} {tg:>+10.1f} {tf:>9.2f} {tn:>+10.1f}")

    if apply:
        conn.executemany(
            "UPDATE wallet_trades SET position_pnl_usd=?, prio_bribe_fee_usd=?, "
            "net_pnl_usd=?, position_pnl_sol=? WHERE trade_id=?", updates)
        conn.commit()
        print(f"APPLIED: updated {len(updates)} rows.")
    else:
        print("DRY RUN: no rows written. Re-run with --apply to persist.")
    conn.close()
    return {"legs": len(updates), "gross_usd": tg, "net_usd": tn, "fees_usd": tf, "applied": apply}


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute historical USD PnL + fees")
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--apply", action="store_true", help="write the recomputed values (default: dry run)")
    args = ap.parse_args()
    if "position_pnl_usd" not in _columns(args.db):
        sys.exit("wallet_trades has no position_pnl_usd column — deploy the schema change first "
                 "(restart the collector, or run `python -m collector.schema` to confirm DDL).")
    recompute(args.db, apply=args.apply)


def _columns(db_path: str) -> set[str]:
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {r[1] for r in c.execute("PRAGMA table_info(wallet_trades)")}
    finally:
        c.close()


if __name__ == "__main__":
    main()
