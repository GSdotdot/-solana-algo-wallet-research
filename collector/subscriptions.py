"""Subscription lifecycle manager.

When a tracked wallet first touches a token: register it for tracking and fire the
one-time REST enrichment burst (in the background, so trade handling isn't blocked).
There is no per-token WS subscription for most tokens: on-chain stats come from the
REST stats poller, and price from the REST batch price call. (An earlier token-screener
stream was retired; identity now comes from the token-stats poll.)

Dedicated price stream: tokens held by a "B" wallet (the fast-scalper subset,
config.B_WALLET_LABELS) instead get their own price socket, because the provider's
price channel is single-active per connection. The socket opens on such a wallet's
entry, is held through the position + B_PRICE_POST_EXIT_SEC after the last B wallet
exits, then closes (falling back to the batch price if another wallet still holds it).

When the last tracked holder exits: start a grace timer sized by today's trade
activity (config.pick_grace_tier). Re-entry cancels it. On expiry, or after the
no-trade hard ceiling, unsubscribe and tear down per-token state.

poll_tier() classifies each token full|coarse for the stats poller's cadence; a
just-exited token stays "full" for POST_EXIT_FULL_SEC, then drops to "coarse".

On reconnect each per-wallet socket re-subscribes its own wallet, and each
targeted-B price socket re-subscribes its own token (both channels are
single-active per connection, so one socket per entity).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from . import config, stats_parser
from .db import Database
from .util import get_logger, now_unix

log = get_logger("subs")


@dataclass
class _TokenSub:
    subscribed_at: int
    last_wallet_trade_at: int
    grace_tier: str | None = None
    grace_expires_at: int | None = None
    b_price_expires_at: int | None = None   # targeted-B price-socket post-exit close time


class SubscriptionManager:
    def __init__(self, db: Database, rest, positions, features, latest_stats,
                 archive, price_sockets, price_socket_wallets):
        self._db = db
        self._rest = rest
        self._positions = positions
        self._features = features
        self._latest_stats = latest_stats
        self._archive = archive
        self._price_sockets = price_sockets      # PriceSocketPool (dedicated price stream)
        # Wallets whose held tokens get a dedicated price socket + exclusion from the
        # batch price call: the targeted-B scalpers (config).
        self._price_socket_wallets = set(price_socket_wallets)
        self._tokens: dict[str, _TokenSub] = {}
        self._discovering: set[str] = set()
        self._bg: set[asyncio.Task] = set()
        self._stop = asyncio.Event()

    # -- registry queries (used by the archive loop) -------------------------
    def is_active(self, token: str) -> bool:
        return token in self._tokens

    def subscribed_at(self, token: str) -> int | None:
        sub = self._tokens.get(token)
        return sub.subscribed_at if sub else None

    def active_tokens(self) -> list[str]:
        return list(self._tokens)

    def batch_price_tokens(self) -> list[str]:
        """Active tokens the batch price call should price: every active token
        EXCEPT those served by a dedicated targeted-B price socket."""
        priced = self._price_sockets.priced_tokens()
        return [t for t in self._tokens if t not in priced]

    def poll_tier(self, token: str) -> str:
        """Stats-poll cadence tier: 'full' while a tracked wallet holds the token, or
        within POST_EXIT_FULL_SEC of its last trade (the just-exited tail we still want
        densely sampled); otherwise 'coarse' (deep grace: cheap upkeep)."""
        if self._positions.holders(token):
            return "full"
        sub = self._tokens.get(token)
        if sub is not None and now_unix() - sub.last_wallet_trade_at <= config.POST_EXIT_FULL_SEC:
            return "full"
        return "coarse"

    # -- discovery -----------------------------------------------------------
    async def on_first_touch(self, token: str, wallet: str, ts: int) -> None:
        if token in self._tokens or token in self._discovering:
            return
        self._discovering.add(token)
        self._tokens[token] = _TokenSub(subscribed_at=ts, last_wallet_trade_at=ts)
        await self._persist(token)
        # Persist discovery provenance NOW via upsert, so the token_stats identity
        # fill (or a REST blob) landing first cannot lose discovered_via/first_seen.
        await self._db.upsert("token_identity", {
            "token_address": token,
            "first_seen_timestamp": ts,
            "discovered_via_wallet": wallet,
        })
        log.info("tracking token %s (discovered via %s)", token, wallet)
        # REST enrichment runs in the background: it must not block ingestion.
        self._spawn(self._enrich(token, wallet, ts))
        self._discovering.discard(token)

    async def _enrich(self, token: str, wallet: str, ts: int) -> None:
        try:
            blobs = await self._rest.burst(token, ts)
        except Exception:  # noqa: BLE001
            log.exception("REST burst failed for %s", token)
            return
        # Upsert only the enrichment blobs onto the already-created identity row;
        # never INSERT OR IGNORE (that would drop them whenever the row already exists).
        await self._db.upsert("token_identity", {"token_address": token, **blobs})
        # One-time bonding-curve state (graduated/progress/reserves). The launchpad
        # detail is static/slow, so it's fetched ONCE here, not on the recurring poll.
        try:
            md = stats_parser.from_launchpad_detail(await self._rest.launchpad_detail(token))
            if md:
                self._latest_stats[token] = {**self._latest_stats.get(token, {}), **md}
        except Exception:  # noqa: BLE001
            log.exception("launchpad detail fetch failed for %s", token)

    # -- per-trade updates ---------------------------------------------------
    async def note_trade(self, token: str, wallet: str, ts: int) -> None:
        sub = self._tokens.get(token)
        if sub is None:
            return
        sub.last_wallet_trade_at = ts
        holders = self._positions.holders(token)
        if holders:
            sub.grace_tier = None
            sub.grace_expires_at = None
        else:
            trades_today = self._positions.trades_today(token, ts)
            last_hold = self._positions.last_hold(token)
            tier, secs = config.pick_grace_tier(trades_today, last_hold)
            sub.grace_tier = tier
            sub.grace_expires_at = ts + secs
            log.info("token %s: last holder exited -> grace %s (%ds)", token, tier, secs)
        self._manage_b_price(token, sub, ts)
        await self._persist(token)

    # -- targeted-B price sockets --------------------------------------------
    def _manage_b_price(self, token: str, sub: _TokenSub, ts: int) -> None:
        """Open/extend this token's dedicated price socket. Open (idempotent) while
        a B-wallet holds it; when the last B-wallet exits, arm a B_PRICE_POST_EXIT_SEC
        timer the lifecycle loop closes on expiry. Re-entry clears the timer + reopens."""
        if not self._price_socket_wallets:
            return
        if self._positions.holders(token) & self._price_socket_wallets:
            sub.b_price_expires_at = None
            self._price_sockets.open(token)
        elif self._price_sockets.is_priced(token) and sub.b_price_expires_at is None:
            sub.b_price_expires_at = ts + config.B_PRICE_POST_EXIT_SEC

    async def _close_b_price(self, token: str, sub: _TokenSub) -> None:
        """Post-exit teardown of a B price socket (token stays subscribed; price falls
        back to the batch price). Skipped if a B-wallet re-entered during the tail."""
        sub.b_price_expires_at = None
        if self._positions.holders(token) & self._price_socket_wallets:
            return
        await self._price_sockets.close(token)
        log.info("token %s: targeted-B price socket closed (post-exit) -> batch price", token)

    # -- lifecycle loop ------------------------------------------------------
    async def run(self) -> None:
        log.info("subscription lifecycle loop started")
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=config.SNAPSHOT_INTERVAL_SEC)
            except asyncio.TimeoutError:
                pass
            if self._stop.is_set():
                break
            await self._tick()
        log.info("subscription lifecycle loop stopped")

    async def _tick(self) -> None:
        now = now_unix()
        for token, sub in list(self._tokens.items()):
            if sub.grace_expires_at is not None and now >= sub.grace_expires_at:
                await self.unsubscribe(token, reason=f"grace {sub.grace_tier} expired")
                continue
            if now - sub.last_wallet_trade_at >= config.HARD_CEILING_SEC:
                await self.unsubscribe(token, reason="no-trade hard ceiling")
                continue
            # token stays subscribed; only its dedicated B price socket times out
            if sub.b_price_expires_at is not None and now >= sub.b_price_expires_at:
                await self._close_b_price(token, sub)

    async def unsubscribe(self, token: str, reason: str) -> None:
        if token not in self._tokens:
            return
        # No per-token stats stream to unsubscribe. Close the targeted-B price
        # socket if this token had one (no-op otherwise).
        await self._price_sockets.close(token)
        self._tokens.pop(token, None)
        self._features.reset(token)
        self._latest_stats.pop(token, None)
        self._archive.last_snapshot_id.pop(token, None)
        await self._db.execute(
            "UPDATE subscription_state SET screener_active=0, price_active=0, "
            "grace_tier=NULL, grace_expires_at=NULL WHERE token_address=?", (token,))
        log.info("unsubscribed token %s (%s)", token, reason)

    # -- persistence ---------------------------------------------------------
    async def _persist(self, token: str) -> None:
        sub = self._tokens[token]
        holders = sorted(self._positions.holders(token))
        # screener_active=0: the token-screener stream is retired (identity now comes from
        # the token-stats poll). price_active=1: price always flows, via either the
        # batch price call or a targeted-B socket.
        await self._db.execute(
            "INSERT INTO subscription_state "
            "(token_address, subscribed_at, screener_active, price_active, "
            " last_wallet_trade_at, wallets_holding, grace_tier, grace_expires_at) "
            "VALUES (?,?,0,1,?,?,?,?) "
            "ON CONFLICT(token_address) DO UPDATE SET "
            " screener_active=0, price_active=1, "
            " last_wallet_trade_at=excluded.last_wallet_trade_at, "
            " wallets_holding=excluded.wallets_holding, "
            " grace_tier=excluded.grace_tier, "
            " grace_expires_at=excluded.grace_expires_at",
            (token, sub.subscribed_at, sub.last_wallet_trade_at,
             json.dumps(holders), sub.grace_tier, sub.grace_expires_at))

    # -- shutdown ------------------------------------------------------------
    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def stop(self) -> None:
        self._stop.set()

    async def drain(self) -> None:
        """Await outstanding background enrichment tasks (graceful shutdown)."""
        if self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)
