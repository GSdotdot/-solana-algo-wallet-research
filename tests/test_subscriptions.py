"""Subscription lifecycle (collector/subscriptions.py): the targeted-B price-socket
open / hold / post-exit close, re-entry cancel, cadence tiers and grace tiers.

Offline: drives a real PositionBook through the manager with stubbed I/O and a
controllable clock. No network, no API credits.
"""

import asyncio

from collector import config, subscriptions
from collector.positions import PositionBook
from collector.subscriptions import SubscriptionManager

B, NB, TOK = "Bwallet", "NBwallet", "Tok1"
PEX = config.B_PRICE_POST_EXIT_SEC


class _Pool:   # mirrors PriceSocketPool: open() is idempotent
    def __init__(self):
        self.opened, self.closed, self.live = [], [], set()

    def is_priced(self, t):
        return t in self.live

    def priced_tokens(self):
        return set(self.live)

    def open(self, t):
        if t in self.live:
            return
        self.opened.append(t)
        self.live.add(t)

    async def close(self, t):
        if t in self.live:
            self.closed.append(t)
            self.live.discard(t)


class _DB:
    async def execute(self, *a, **k):
        return 0

    async def upsert(self, *a, **k):
        return None


class _Rest:
    async def burst(self, token, ts):
        return {}

    async def launchpad_detail(self, token):
        return None


class _Arch:
    def __init__(self):
        self.last_snapshot_id = {}


class _Feat:
    def reset(self, t):
        pass


def _mk(pos, pool):
    return SubscriptionManager(_DB(), _Rest(), pos, _Feat(), {}, _Arch(), pool, {B})


def _clock(monkeypatch, start):
    clock = {"t": start}
    monkeypatch.setattr(subscriptions, "now_unix", lambda: clock["t"])
    return clock


def test_b_price_socket_lifecycle(monkeypatch):
    """Open on B-entry, hold through a non-B co-holder, close only after the post-exit
    window, then fall back to the batch price; cadence tier drops to coarse in grace."""
    clock = _clock(monkeypatch, 1_000_000)

    async def run():
        pos, pool = PositionBook(), _Pool()
        subs = _mk(pos, pool)
        t = clock["t"]
        pos.on_trade(B, TOK, "BUY", 1000.0, 1.0, 64.0, t)
        await subs.on_first_touch(TOK, B, t)
        await subs.note_trade(TOK, B, t)
        assert pool.opened == [TOK]
        assert subs.poll_tier(TOK) == "full"
        assert subs.batch_price_tokens() == []                 # B-priced -> excluded

        pos.on_trade(NB, TOK, "BUY", 500.0, 0.5, 32.0, t + 5)  # non-B co-holder
        await subs.note_trade(TOK, NB, t + 5)
        assert pool.opened == [TOK] and pool.is_priced(TOK)

        pos.on_trade(B, TOK, "SELL", 1000.0, 1.2, 77.0, t + 10)   # last B exits (NB still in)
        await subs.note_trade(TOK, B, t + 10)
        assert pool.closed == [], "must not close inside the post-exit window"
        assert subs._tokens[TOK].b_price_expires_at == t + 10 + PEX

        clock["t"] = t + 10 + PEX - 1
        await subs._tick()
        assert pool.closed == [], "closed too early"

        clock["t"] = t + 10 + PEX + 1
        await subs._tick()
        assert pool.closed == [TOK]
        assert subs.batch_price_tokens() == [TOK]              # NB still holds -> batch price
        assert subs.poll_tier(TOK) == "full"

        pos.on_trade(NB, TOK, "SELL", 500.0, 0.4, 26.0, clock["t"])   # NB exits -> grace
        await subs.note_trade(TOK, NB, clock["t"])
        assert subs._tokens[TOK].grace_tier is not None
        assert subs.poll_tier(TOK) == "full"                   # within POST_EXIT_FULL_SEC
        clock["t"] += config.POST_EXIT_FULL_SEC + 1
        assert subs.poll_tier(TOK) == "coarse"                 # deep grace
        await subs.drain()

    asyncio.run(run())


def test_b_reentry_cancels_the_close(monkeypatch):
    clock = _clock(monkeypatch, 2_000_000)

    async def run():
        pos, pool = PositionBook(), _Pool()
        subs = _mk(pos, pool)
        t = clock["t"]
        pos.on_trade(B, TOK, "BUY", 1000.0, 1.0, 64.0, t)
        await subs.on_first_touch(TOK, B, t)
        await subs.note_trade(TOK, B, t)
        pos.on_trade(B, TOK, "SELL", 1000.0, 1.1, 70.0, t + 5)
        await subs.note_trade(TOK, B, t + 5)
        assert subs._tokens[TOK].b_price_expires_at is not None
        pos.on_trade(B, TOK, "BUY", 800.0, 0.9, 58.0, t + 10)     # re-entry
        await subs.note_trade(TOK, B, t + 10)
        assert subs._tokens[TOK].b_price_expires_at is None
        assert pool.is_priced(TOK) and pool.closed == []
        await subs.drain()

    asyncio.run(run())


def test_non_b_token_never_opens_a_socket(monkeypatch):
    clock = _clock(monkeypatch, 3_000_000)

    async def run():
        pos, pool = PositionBook(), _Pool()
        subs = _mk(pos, pool)
        t = clock["t"]
        pos.on_trade(NB, "Tok2", "BUY", 1000.0, 1.0, 64.0, t)
        await subs.on_first_touch("Tok2", NB, t)
        await subs.note_trade("Tok2", NB, t)
        assert pool.opened == [] and subs.batch_price_tokens() == ["Tok2"]
        await subs.drain()

    asyncio.run(run())


def test_grace_expiry_and_hard_ceiling_unsubscribe(monkeypatch):
    clock = _clock(monkeypatch, 4_000_000)

    async def run():
        pos, pool = PositionBook(), _Pool()
        subs = _mk(pos, pool)
        t = clock["t"]
        # token 1: round trip -> grace -> expires
        pos.on_trade(NB, "G", "BUY", 10.0, 0.1, 6.0, t)
        await subs.on_first_touch("G", NB, t)
        await subs.note_trade("G", NB, t)
        pos.on_trade(NB, "G", "SELL", 10.0, 0.1, 6.0, t + 1)
        await subs.note_trade("G", NB, t + 1)
        expires = subs._tokens["G"].grace_expires_at
        # token 2: still held, but no trade for longer than the hard ceiling
        pos.on_trade(NB, "H", "BUY", 10.0, 0.1, 6.0, t)
        await subs.on_first_touch("H", NB, t)
        await subs.note_trade("H", NB, t)

        clock["t"] = expires + 1
        await subs._tick()
        assert not subs.is_active("G") and subs.is_active("H")
        clock["t"] = t + config.HARD_CEILING_SEC + 1
        await subs._tick()
        assert not subs.is_active("H")
        await subs.drain()

    asyncio.run(run())


def test_pick_grace_tier():
    tiers = config.GRACE_TIERS
    frequent, quick = config.FREQUENT_TRADES_TODAY, config.QUICK_HOLD_SEC
    assert config.pick_grace_tier(frequent, quick - 1) == ("long", tiers["long"])      # frequent wins
    assert config.pick_grace_tier(1, quick - 1) == ("short", tiers["short"])
    assert config.pick_grace_tier(1, quick) == ("medium", tiers["medium"])
    assert config.pick_grace_tier(1, None) == ("medium", tiers["medium"])
