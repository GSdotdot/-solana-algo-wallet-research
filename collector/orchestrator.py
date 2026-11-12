"""Top-level wiring + message routing + graceful shutdown.

Owns the shared in-memory cache (latest_stats) and constructs all components,
then routes each inbound WS frame:
  wallet-transaction frame -> TradeProcessor                 (ingest, discover, label)
  price-bar frame          -> tick archive + latest_stats['price']  (targeted-B sockets)
WS sockets: one per wallet (the wallet-transaction channel is single-active per
connection) + a dynamic pool of one per targeted-B token (the price channel is
single-active too). Token identity comes from the token-stats poll
(stats_poller on_identity callback). Most tokens' price comes from the REST batch
price call; targeted-B tokens get the dedicated socket above.

On SIGINT/SIGTERM: stop the loops, drain background enrichment, flush the snapshot
+ tick buffers, close REST/WS/DB, and log the API-credit total.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from typing import Any

from common.params import param

from . import config
from .credits import CreditTracker
from .db import Database
from .features import FeatureEngine
from .labeler import EventLabeler
from .positions import PositionBook
from .price_sockets import PriceSocketPool
from .rest_client import RestClient
from .snapshot_archive import SnapshotArchive
from .stats_parser import identity_from_token_stats
from .stats_poller import StatsPoller
from .subscriptions import SubscriptionManager
from .tick_archive import TickArchive
from .util import get_logger
from .wallet_txs import TradeProcessor
from .ws_client import WSClient

log = get_logger("orch")


class Orchestrator:
    def __init__(self) -> None:
        self.credits = CreditTracker()
        self.db = Database(config.DB_PATH)
        self.latest_stats: dict[str, dict[str, Any]] = {}   # filled by StatsPoller (REST + price)
        self._identity_filled: set[str] = set()
        self._dex_map = config.load_dex_programs()
        self.wallets = config.load_wallets()
        self.b_wallets = config.b_wallet_addresses(self.wallets)
        self.wallet_fees = config.load_wallet_fees()   # {addr: {buy_usd, sell_usd}} priority+bribe
        self._unknown_frames: dict[str, int] = {}
        # Diagnostic: WRE_LOG_FRAMES=1 logs the first sample + running count of
        # every inbound frame type.
        self._log_frames = bool(os.environ.get("WRE_LOG_FRAMES"))
        self._frame_counts: dict[str, int] = {}
        self._frame_samples: set[str] = set()

        self.features = FeatureEngine()
        self.positions = PositionBook()
        self.rest = RestClient(self.credits)
        self.price_sockets = PriceSocketPool(self._on_message)   # targeted-B price sockets
        self.wallet_ws: list[WSClient] = []   # one socket per wallet, built in run()
        self.tick_archive = TickArchive(self.db) if config.STORE_TICKS else None
        self.archive = SnapshotArchive(self.db, self.features, None, self.latest_stats)
        self.subs = SubscriptionManager(
            self.db, self.rest, self.positions, self.features,
            self.latest_stats, self.archive, self.price_sockets, self.b_wallets)
        self.archive._subs = self.subs  # resolve the archive<->subs cycle
        self.stats_poller = StatsPoller(
            self.rest, self.subs, self.latest_stats, self.credits,
            on_identity=self._on_token_stats_identity)
        self.labeler = EventLabeler(self.db, self.archive)
        self.trade_proc = TradeProcessor(
            self.db, self.subs, self.positions, self.labeler, self.archive,
            self._dex_map, self.stats_poller, self.wallet_fees)

    # -- routing -------------------------------------------------------------
    async def _on_message(self, msg: dict[str, Any]) -> None:
        if not isinstance(msg, dict):
            return
        mtype = msg.get(config.MSG_TYPE_KEY)
        if self._log_frames:
            self._frame_counts[mtype] = self._frame_counts.get(mtype, 0) + 1
            if mtype not in self._frame_samples:
                self._frame_samples.add(mtype)
                log.info("FIRST %s frame: %s", mtype, json.dumps(msg)[:900])
        if mtype == config.MSG_WALLET_TXS:
            await self.trade_proc.handle(msg.get(config.MSG_DATA_KEY) or {})
        elif mtype == config.MSG_PRICE:
            self._on_price(msg.get(config.MSG_DATA_KEY) or {})
        else:
            self._unknown_frames[mtype] = self._unknown_frames.get(mtype, 0) + 1
            log.debug("unhandled frame type=%s", mtype)

    def _on_price(self, data: dict[str, Any]) -> None:
        """A targeted-B price bar: archive the raw bar and feed the latest USD close
        into latest_stats['price'] (the archive folds it into token_snapshots.price;
        the labeler reads it for exact-moment trade context)."""
        fields = param("data_provider.response_fields.price_bar")
        token = data.get(fields["token"])
        if not token:
            return
        if self.tick_archive is not None:
            self.tick_archive.record(token, data)   # raw bar (no REST credits)
        close = data.get(fields["close"])
        # Only feed a still-tracked token, and never resurrect a popped entry (a late
        # frame arriving after unsubscribe must not recreate latest_stats).
        if close is None or not self.subs.is_active(token):
            return
        try:
            self.latest_stats[token] = {**self.latest_stats.get(token, {}), "price": float(close)}
        except (TypeError, ValueError):
            pass

    async def _on_token_stats_identity(self, token: str, resp: dict[str, Any] | None) -> None:
        """Fill token_identity from a token-stats poll, once per token
        (name/symbol/decimals/logo/socials)."""
        if token in self._identity_filled:
            return
        identity = identity_from_token_stats(resp)
        if not identity.get("token_name") and not identity.get("token_symbol"):
            return  # token-stats not ready / partial: retry on the next poll
        self._identity_filled.add(token)
        await self._upsert_identity({**identity, "token_address": token})

    async def _upsert_identity(self, identity: dict[str, Any]) -> None:
        token = identity.get("token_address")
        if not token:
            return
        # Write only known (non-null) fields. Discovery provenance and REST blobs are
        # owned by the subscription path; upsert won't clobber them.
        cols = {k: v for k, v in identity.items() if v is not None}
        cols["token_address"] = token
        await self.db.upsert("token_identity", cols)

    def _make_wallet_oc(self, wallet: str):
        """on_connect for a per-wallet socket: subscribe that one wallet via that
        exact socket (the channel is single-active per connection)."""
        async def _oc(ws) -> None:
            await ws.send(config.sub_wallet(wallet))
        return _oc

    def _build_wallet_sockets(self, wallets: list[dict]) -> list[WSClient]:
        # Sockets are named by label, never by address, so no address reaches the log.
        return [WSClient(self._on_message, self._make_wallet_oc(w["address"]),
                         name=f"w:{w.get('label') or i}") for i, w in enumerate(wallets, 1)]

    # -- lifecycle -----------------------------------------------------------
    async def run(self) -> None:
        await self.db.connect()
        self.wallet_ws = self._build_wallet_sockets(self.wallets)
        log.info("loaded %d wallets (%d wallet WS sockets), %d DEX programs; db=%s",
                 len(self.wallets), len(self.wallet_ws), len(self._dex_map), config.DB_PATH.name)
        if self.b_wallets:
            b_labels = sorted(w.get("label") or "?" for w in self.wallets
                              if w["address"] in self.b_wallets)
            log.info("targeted-B: %d wallet(s) %s -> dedicated price sockets",
                     len(self.b_wallets), b_labels)
        else:
            log.warning("targeted-B: 0 wallets resolved from labels %s; NO dedicated price "
                        "sockets will open (every token uses the batch price call)",
                        sorted(config.B_WALLET_LABELS))
        n_fee = sum(1 for w in self.wallets if w["address"] in self.wallet_fees)
        log.info("priority+bribe fees configured for %d/%d wallets -> net_pnl_usd%s",
                 n_fee, len(self.wallets), "" if n_fee else " (none: net == gross until set)")

        stop = asyncio.Event()
        self._install_signals(stop)

        # Smoke-run knob: WRE_RUN_SECONDS=N stops the run cleanly after N seconds
        # (sets the same stop event signals use), so a bounded smoke test flushes
        # buffers + logs credits without relying on Ctrl+C.
        run_seconds = os.environ.get("WRE_RUN_SECONDS")
        if run_seconds:
            log.info("auto-stop armed: %ss (bounded run)", run_seconds)
            asyncio.get_running_loop().call_later(int(run_seconds), stop.set)

        tasks = [
            asyncio.create_task(self.archive.run(), name="archive"),
            asyncio.create_task(self.subs.run(), name="subs"),
            asyncio.create_task(self.stats_poller.run(), name="stats"),
            asyncio.create_task(self.stats_poller.run_price(), name="price"),
        ]
        if self.tick_archive is not None:
            tasks.append(asyncio.create_task(self.tick_archive.run(), name="ticks"))
        # targeted-B price sockets are spawned on demand by PriceSocketPool, not here.
        tasks += [asyncio.create_task(w.run(), name=w.name) for w in self.wallet_ws]
        try:
            await stop.wait()
        finally:
            await self._shutdown(tasks)

    def _install_signals(self, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                # Windows: add_signal_handler isn't supported for the loop.
                signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))

    async def _shutdown(self, tasks: list[asyncio.Task]) -> None:
        log.info("graceful shutdown initiated")
        self.stats_poller.stop()
        self.archive.stop()
        self.subs.stop()
        for w in self.wallet_ws:
            await w.close()
        # Close all price-bar sources (the targeted-B sockets) BEFORE the tick sink,
        # so the tick archive's final flush captures every buffered bar.
        await self.price_sockets.close_all()
        if self.tick_archive is not None:
            self.tick_archive.stop()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.subs.drain()        # finish in-flight REST enrichment
        await self.stats_poller.drain()  # finish in-flight at-trade polls
        await self.archive.flush()     # final snapshot buffer flush
        await self.rest.close()
        await self.db.close()
        if self._unknown_frames:
            log.info("unhandled frame types seen: %s", dict(self._unknown_frames))
        if self._log_frames:
            log.info("ALL frame-type counts: %s", dict(self._frame_counts))
        log.info("API-credit summary: %s", json.dumps(self.credits.summary()))
        log.info("shutdown complete")
