"""Raw 1-second price-bar archive, scoped to the targeted-B price sockets.

The targeted-B price sockets stream one OHLCV bar per second, but token_snapshots
only samples `price` at the (coarser) snapshot cadence. This persists every bar to
`price_ticks` for fine-grained batch work (indicator sweeps, exit-path
reconstruction for the fast scalpers). It costs no REST credits, only disk/IO.

Naming note: the table is called `price_ticks` for historical reasons, but each row
is a 1-second BAR (open/high/low/close/volume), not an individual trade. The only
true per-event data in this project is `wallet_trades` (one row per tracked-wallet
transaction).

record() is called synchronously from the price-frame handler (buffer append). A
background loop flushes the buffer (OR REPLACE so a bar updated mid-second keeps
its final value) and periodically prunes rows older than TICK_RETENTION_HRS: a
rolling window sized to outlast the periodic export so nothing is lost before pruning.
"""

from __future__ import annotations

import asyncio
from typing import Any

from common.params import param

from . import config
from .db import Database
from .util import get_logger, now_unix

log = get_logger("ticks")


class TickArchive:
    def __init__(self, db: Database, retention_hrs: int | None = None) -> None:
        self._db = db
        self._retention_sec = (retention_hrs or config.TICK_RETENTION_HRS) * 3600
        self._buf: list[dict[str, Any]] = []
        self._stop = asyncio.Event()
        self._last_flush = now_unix()
        self._last_prune = now_unix()
        self._written = 0

    def record(self, token: str, data: dict[str, Any]) -> None:
        """Buffer one price frame as a 1s bar. Sync (called from the WS handler)."""
        F = param("data_provider.response_fields.price_bar")
        ut = data.get(F["bar_time"])
        if not token or ut is None:
            return
        self._buf.append({
            "token_address": token,
            "unix_time": ut,
            "ohlcv_open": data.get(F["open"]), "ohlcv_high": data.get(F["high"]),
            "ohlcv_low": data.get(F["low"]), "ohlcv_close": data.get(F["close"]),
            "volume": data.get(F["volume"]), "volume_usd": data.get(F["volume_usd"]),
            "received_at": now_unix(),
        })

    async def run(self) -> None:
        log.info("tick archive started (retention %dh)", self._retention_sec // 3600)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=config.TICK_FLUSH_INTERVAL_SEC)
            except asyncio.TimeoutError:
                pass
            if self._stop.is_set():
                break
            await self.flush()
            now = now_unix()
            if now - self._last_prune >= config.TICK_PRUNE_INTERVAL_SEC:
                await self._prune()
                self._last_prune = now
        await self.flush()
        log.info("tick archive stopped (%d bars written)", self._written)

    async def flush(self) -> None:
        if not self._buf:
            self._last_flush = now_unix()
            return
        batch, self._buf = self._buf, []
        # OR REPLACE: a 1s bar that updates within its second keeps the final value.
        await self._db.insert_many("price_ticks", batch, or_replace=True)
        self._written += len(batch)
        self._last_flush = now_unix()

    async def _prune(self) -> None:
        cutoff = now_unix() - self._retention_sec
        n = await self._db.execute("DELETE FROM price_ticks WHERE unix_time < ?", (cutoff,))
        if n:
            log.info("pruned %d bars older than the retention window", n)

    def stop(self) -> None:
        self._stop.set()
