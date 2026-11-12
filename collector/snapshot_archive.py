"""Periodic token snapshot archive (Dataset 2) + exact-moment context provider.

Every SNAPSHOT_INTERVAL_SEC, for each actively tracked token, take the latest
cached stats (REST polls + price) as one token_snapshots row, run Module A
features against it (committing the per-token baseline), and buffer the row.
Buffers flush at BUFFER_SIZE rows or FLUSH_INTERVAL_SEC, whichever comes first,
and on shutdown.

context_for_trade() builds the SAME merged vector at an arbitrary instant WITHOUT
committing the feature baseline. This is the exact-moment stamp the labeler puts
on labeled_events, so a trade's context is precise to the moment, not rounded to
a snapshot boundary. last_snapshot_id links the event to the nearest archived row.

The cache (latest_stats) is owned by the orchestrator and shared by reference; the
REST stats/price pollers and the targeted-B price sockets keep it fresh.
"""

from __future__ import annotations

import asyncio
from typing import Any

from . import config
from .db import Database
from .features import FeatureEngine
from .util import get_logger, now_unix

log = get_logger("archive")


class SnapshotArchive:
    def __init__(
        self,
        db: Database,
        features: FeatureEngine,
        subs: Any,                               # SubscriptionManager (duck-typed)
        latest_stats: dict[str, dict[str, Any]],
    ) -> None:
        self._db = db
        self._fe = features
        self._subs = subs
        self._latest_stats = latest_stats
        self._buffer: list[dict[str, Any]] = []
        self._last_flush = now_unix()
        self._stop = asyncio.Event()
        self.last_snapshot_id: dict[str, str] = {}

    async def run(self) -> None:
        log.info("archive loop started (%ss cadence)", config.SNAPSHOT_INTERVAL_SEC)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=config.SNAPSHOT_INTERVAL_SEC)
            except asyncio.TimeoutError:
                pass
            if self._stop.is_set():
                break
            await self._tick()
        await self.flush()
        log.info("archive loop stopped")

    async def _tick(self) -> None:
        ts = now_unix()
        for token in self._subs.active_tokens():
            row = self._build_row(token, ts, commit=True)
            if row is None:
                continue  # no stats poll has produced a value for this token yet
            self._buffer.append(row)
            self.last_snapshot_id[token] = row["snapshot_id"]
        await self._maybe_flush(ts)

    def _merge(self, token: str, ts: int, commit: bool) -> dict[str, Any] | None:
        """Cached stats (incl. `price`) + Module A features.

        `price` is the per-token price series for every active token (batch price
        call, or the dedicated socket for targeted-B tokens); the `ohlcv_*` snapshot
        columns are no longer written. Returns None only when no stats poll has
        produced a value yet (e.g. a just-discovered token before its first poll)."""
        stats = self._latest_stats.get(token)
        if not stats:
            return None
        merged: dict[str, Any] = dict(stats)                  # scalars + 48 windows + price
        merged.update(self._fe.compute(token, stats, ts, commit=commit))
        return merged

    def _build_row(self, token: str, ts: int, commit: bool) -> dict[str, Any] | None:
        merged = self._merge(token, ts, commit)
        if merged is None:
            return None
        sub_at = self._subs.subscribed_at(token)
        return {
            "snapshot_id": f"{token}:{ts}",
            "token_address": token,
            "timestamp": ts,
            "seconds_since_subscription": (ts - sub_at) if sub_at else None,
            **merged,
        }

    async def _maybe_flush(self, ts: int) -> None:
        if (len(self._buffer) >= config.BUFFER_SIZE
                or ts - self._last_flush >= config.FLUSH_INTERVAL_SEC):
            await self.flush()

    async def flush(self) -> None:
        if not self._buffer:
            self._last_flush = now_unix()
            return
        batch, self._buffer = self._buffer, []
        n = await self._db.insert_many("token_snapshots", batch)
        self._last_flush = now_unix()
        log.debug("flushed %d snapshots (%d new)", len(batch), n)

    # -- exact-moment context for the labeler --------------------------------
    def context_for_trade(self, token: str) -> dict[str, Any]:
        """Merged stats + derived vector for the latest cached state, read-only
        (no baseline commit). Empty only if nothing has been polled yet.

        Features are computed at now_unix(), the SAME wall clock the archive
        commits its baseline with, so the at-trade velocity dt is not measured
        against the trade's block time (a different clock), which would
        otherwise null or mis-scale every velocity on labeled_events."""
        return self._merge(token, now_unix(), commit=False) or {}

    def stop(self) -> None:
        self._stop.set()
