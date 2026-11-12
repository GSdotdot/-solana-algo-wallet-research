"""Module B — event labeler.

On every tracked-wallet trade that carries a position label (ENTRY / EXIT / ADD
/ PARTIAL_EXIT / FLIP), write one self-contained labeled_events row: the exact
on-chain context at the trade instant (from the archive's read-only merge: token
scalars + 48 window fields + Module A features) plus the trade metadata.

The full self-contained copy means the modelling matrix needs no joins; the
snapshot_id still links to the nearest archived snapshot for time-series scroll-back.
Position math itself lives in positions.PositionBook — this module only persists.

db.insert maps dict keys to columns and ignores extras, so passing the merged
context dict straight through is safe even though it contains snapshot-only keys.
"""

from __future__ import annotations

from typing import Any

from .db import Database
from .snapshot_archive import SnapshotArchive
from .util import get_logger

log = get_logger("labeler")

_TRADE_META = ("direction", "amount_sol", "amount_token", "price_per_token", "volume_usd")


class EventLabeler:
    def __init__(self, db: Database, archive: SnapshotArchive) -> None:
        self._db = db
        self._archive = archive

    async def on_trade(
        self,
        trade_row: dict[str, Any],
        label: str,
        token: str,
        ts: int,
        has_pre_context: bool,
        pre_context_duration_sec: int,
    ) -> None:
        ctx = self._archive.context_for_trade(token)  # {} if nothing polled yet

        row: dict[str, Any] = {
            "event_id": f"{trade_row['trade_id']}:{label}",
            "trade_id": trade_row["trade_id"],
            "snapshot_id": self._archive.last_snapshot_id.get(token),
            "wallet_address": trade_row["wallet_address"],
            "token_address": token,
            "timestamp": ts,
            "label": label,
        }
        row.update(ctx)
        for k in _TRADE_META:
            row[k] = trade_row.get(k)
        row["has_pre_context"] = int(has_pre_context)
        row["pre_context_duration_sec"] = pre_context_duration_sec

        await self._db.insert("labeled_events", row)
        log.debug("labeled %s %s on %s (pre_ctx=%s)", label,
                  trade_row.get("wallet_address"), token, has_pre_context)
