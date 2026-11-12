"""Per-token REST stats poller with lifecycle-aware cadence tiers.

Per-token on-chain stats come from REST. This polls each active token and merges
results into the shared `latest_stats` cache the archive folds into its periodic
snapshots (carry-forward between polls: holder / trader-type fields move slowly,
so brief staleness is fine). It also fills token_identity from the token-stats
response (identity_from_token_stats): name/symbol/decimals/logo/socials.

Cadence tiers (subs.poll_tier; see config):
  full  (held, or <= POST_EXIT_FULL_SEC since the last holder exit):
        token stats @ STATS_POLL_INTERVAL_SEC + holder breakdown @ HOLDER_BREAKDOWN_FULL_SEC.
  coarse (grace, > POST_EXIT_FULL_SEC since the last trade):
        token stats @ STATS_POLL_COARSE_SEC, NO background holder breakdown.
An at-trade poll fires in BOTH tiers (exact decision-moment context). The launchpad
detail is NOT polled here: it is fetched ONCE at discovery (subscriptions).

Two API-credit backstops sit on top of the tiers: a pro-rated pace guard (State 2,
sticky: once cumulative REST credits exceed CREDIT_PACE_PER_DAY * elapsed/day after a
warm-up, full-tier token-stats polling is demoted to the coarse cadence) and a hard
ceiling that stops all REST polling entirely. Credits are read from the shared
CreditTracker.
"""

from __future__ import annotations

import asyncio

from . import config, stats_parser
from .credits import CreditTracker
from .util import get_logger, now_unix

log = get_logger("stats")


class StatsPoller:
    def __init__(self, rest, subs, latest_stats: dict, credits: CreditTracker,
                 interval: int | None = None, on_identity=None) -> None:
        self._rest = rest
        self._subs = subs
        self._latest_stats = latest_stats
        self._credits = credits
        self._interval = interval or config.STATS_POLL_INTERVAL_SEC
        # async (token, stats_resp) -> None: fills token_identity from the
        # token-stats poll. Optional.
        self._on_identity = on_identity
        self._stop = asyncio.Event()
        self._start = 0
        self._state2 = False            # sticky once engaged
        self._ceiling_hit = False       # logged once
        self._last_stats_poll: dict[str, int] = {}  # per-token last token_stats poll
        self._last_hp: dict[str, int] = {}      # per-token last holder breakdown poll
        self._last_poll: dict[str, int] = {}     # per-token last any poll (at-trade debounce)
        self._bg: set[asyncio.Task] = set()
        self._rounds = 0

    # -- merge helpers -------------------------------------------------------
    def _merge(self, token: str, stats: dict) -> None:
        if stats:
            self._latest_stats[token] = {**self._latest_stats.get(token, {}), **stats}

    async def _poll_token_stats(self, token: str) -> None:
        resp = await self._rest.token_stats(token)
        self._merge(token, stats_parser.from_token_stats(resp))
        if self._on_identity is not None:
            await self._on_identity(token, resp)

    async def _poll_holder(self, token: str, now: int) -> None:
        self._merge(token, stats_parser.from_holder_breakdown(await self._rest.holder_breakdown(token)))
        self._last_hp[token] = now

    # -- state machine -------------------------------------------------------
    def _over_ceiling(self) -> bool:
        if self._credits.total >= config.CREDIT_HARD_CEILING:
            if not self._ceiling_hit:
                self._ceiling_hit = True
                log.warning("credit hard ceiling reached (%.0f spent); REST polling halted",
                            self._credits.total)
            return True
        return False

    def _in_state2(self, now: int) -> bool:
        if self._state2:
            return True
        elapsed = now - self._start
        if elapsed >= config.THROTTLE_WARMUP_SEC:
            pace_budget = config.CREDIT_PACE_PER_DAY * (elapsed / 86400.0)
            if self._credits.total > pace_budget:
                self._state2 = True
                log.warning(
                    "throttle: State 2 engaged at %.1fh; credits %.0f exceed the pro-rated pace budget %.0f",
                    elapsed / 3600.0, self._credits.total, pace_budget)
        return self._state2

    # -- main loop -----------------------------------------------------------
    async def run(self) -> None:
        self._start = now_unix()
        log.info("stats poller started (token-stats %ss full / %ss coarse, holder %ss full)",
                 self._interval, config.STATS_POLL_COARSE_SEC, config.HOLDER_BREAKDOWN_FULL_SEC)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
            if self._stop.is_set():
                break
            if self._over_ceiling():
                continue
            now = now_unix()
            state2 = self._in_state2(now)
            # under the pace guard, even full-tier token-stats drops to the coarse cadence
            full_ov_cadence = config.STATS_POLL_COARSE_SEC if state2 else self._interval
            for token in self._subs.active_tokens():
                tier = self._subs.poll_tier(token)
                ov_cadence = full_ov_cadence if tier == "full" else config.STATS_POLL_COARSE_SEC
                if now - self._last_stats_poll.get(token, 0) >= ov_cadence:
                    await self._poll_token_stats(token)
                    self._last_stats_poll[token] = now
                    self._last_poll[token] = now
                # holder breakdown: full tier only (coarse/grace skips it to save credits)
                if tier == "full" and now - self._last_hp.get(token, 0) >= config.HOLDER_BREAKDOWN_FULL_SEC:
                    await self._poll_holder(token, now)
            self._rounds += 1
        log.info("stats poller stopped (%d rounds, final state=%s)",
                 self._rounds, "2" if self._state2 else "1")

    # -- batch price loop (ALL active tokens; ceiling-gated, NOT throttled) --
    async def run_price(self) -> None:
        """Poll the batch price call for every active token every
        PRICE_POLL_INTERVAL_SEC and merge `price` into latest_stats (the archive folds
        it into token_snapshots.price for ALL held tokens). Targeted-B tokens are
        EXCLUDED here (batch_price_tokens drops them): they get a dedicated price
        socket instead, which feeds the same `price`. Hard-ceiling-gated only;
        deliberately NOT subject to the State-2 throttle."""
        log.info("price poller started (%ss cadence, batch price; targeted-B excluded)",
                 config.PRICE_POLL_INTERVAL_SEC)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=config.PRICE_POLL_INTERVAL_SEC)
            except asyncio.TimeoutError:
                pass
            if self._stop.is_set():
                break
            if self._over_ceiling():
                continue
            tokens = self._subs.batch_price_tokens()
            if not tokens:
                continue
            try:
                prices = stats_parser.from_batch_price(await self._rest.batch_price(tokens))
            except Exception:  # noqa: BLE001 - a failed poll must never kill the loop or stall ingest
                log.exception("batch_price poll failed; skipping cycle")
                continue
            for token, price in prices.items():
                self._merge(token, {"price": price})
        log.info("price poller stopped")

    # -- at-trade poll (both states) ----------------------------------------
    async def on_trade(self, token: str) -> None:
        """Fire a non-blocking poll for `token` on a wallet trade, in BOTH states.
        A trade can land up to a full background interval after the last poll, so
        this captures the exact decision-moment context. Debounced so it doesn't
        double up with a background poll that just ran."""
        if not config.AT_TRADE_POLL or self._over_ceiling():
            return
        now = now_unix()
        if now - self._last_poll.get(token, 0) < config.AT_TRADE_DEBOUNCE_SEC:
            return  # just polled: skip
        self._last_poll[token] = now
        self._spawn(self._at_trade_poll(token, now))

    async def _at_trade_poll(self, token: str, now: int) -> None:
        try:
            await self._poll_token_stats(token)
            await self._poll_holder(token, now)
        except Exception:  # noqa: BLE001
            log.exception("at-trade poll failed for %s", token)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def stop(self) -> None:
        self._stop.set()

    async def drain(self) -> None:
        if self._bg:
            await asyncio.gather(*list(self._bg), return_exceptions=True)
