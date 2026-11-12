"""Dynamic pool of per-token price WebSocket sockets (targeted-B).

The provider's price channel streams only the last-subscribed token per connection
(confirmed live: the same single-active-per-connection rule as the wallet channel).
So to get a fine-grained price stream for several tokens at once, each token needs
its own socket. This pool opens/closes those sockets on demand as the targeted-B
wallets enter and exit tokens; the lifecycle (open on B-entry, hold + a short
post-exit tail, then close) is driven by SubscriptionManager.

Each socket reuses the name-able WSClient: on every (re)connect it re-subscribes
its one token, and inbound price-bar frames route through the shared on_message
handler (orchestrator._on_price -> tick archive + latest_stats['price']).

Sockets are cheap against the provider's per-key connection limit: the pool only
holds the scalpers' live holdings plus their short post-exit tails.
"""

from __future__ import annotations

import asyncio

from . import config
from .util import get_logger
from .ws_client import WSClient

log = get_logger("px")


class PriceSocketPool:
    def __init__(self, on_message) -> None:
        self._on_message = on_message
        self._sockets: dict[str, WSClient] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    # -- queries -------------------------------------------------------------
    def is_priced(self, token: str) -> bool:
        return token in self._sockets

    def priced_tokens(self) -> set[str]:
        return set(self._sockets)

    def __len__(self) -> int:
        return len(self._sockets)

    # -- lifecycle -----------------------------------------------------------
    def open(self, token: str) -> None:
        """Open a dedicated price socket for `token` (idempotent). Sync: spawns the
        socket's run() task, which connects + subscribes on the running loop."""
        if token in self._sockets:
            return
        ws = WSClient(self._on_message, self._make_on_connect(token), name=f"px:{token[:6]}")
        self._sockets[token] = ws
        self._tasks[token] = asyncio.create_task(ws.run(), name=f"px:{token[:6]}")
        log.info("price socket opened for %s (%d open)", token, len(self._sockets))

    def _make_on_connect(self, token: str):
        async def _oc(ws: WSClient) -> None:
            await ws.send(config.sub_price(token))
        return _oc

    async def close(self, token: str) -> None:
        """Close + drop the socket for `token` (no-op if not open). Awaits the run()
        task so teardown is clean before we fall back to the batch price."""
        ws = self._sockets.pop(token, None)
        task = self._tasks.pop(token, None)
        if ws is not None:
            await ws.close()            # sets stop + closes the live connection
        if task is not None:
            try:
                await task              # let the run() loop observe stop and exit
            except asyncio.CancelledError:
                pass
        if ws is not None:
            log.info("price socket closed for %s (%d open)", token, len(self._sockets))

    async def close_all(self) -> None:
        for token in list(self._sockets):
            await self.close(token)
