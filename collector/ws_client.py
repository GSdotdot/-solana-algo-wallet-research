"""One WebSocket connection to the data provider.

Responsibilities:
  - one named connection per subscribed entity (a wallet, or a targeted-B token):
    the provider's channels are single-active per connection
  - exponential backoff on disconnect, doubling up to a cap
  - protocol-level keepalive ping (websockets built-in)
  - on every (re)connect, call on_connect() so the caller re-subscribes
  - route each inbound frame to on_message(); never let one bad frame kill the loop

websockets 14+: the header kwarg is `additional_headers`. The API key lives only in
the URL built by config.build_ws_url(); we log only the redacted form.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

import websockets

from . import config
from .util import get_logger

log = get_logger("ws")

OnMessage = Callable[[dict[str, Any]], Awaitable[None]]
OnConnect = Callable[[Any], Awaitable[None]]   # receives the WSClient, so it can subscribe via that exact socket


class WSClient:
    def __init__(self, on_message: OnMessage, on_connect: OnConnect, name: str = "ws") -> None:
        self._on_message = on_message
        self._on_connect = on_connect
        self.name = name                 # e.g. "w:<wallet label>", "px:<token prefix>"
        self._ws: Any = None
        self._stop = asyncio.Event()

    async def send(self, msg: dict[str, Any]) -> bool:
        """Send one JSON frame. Returns False if not currently connected
        (the caller relies on on_connect re-subscribe to recover state)."""
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(json.dumps(msg))
            return True
        except Exception as e:  # noqa: BLE001 - connection may have just dropped
            log.warning("send failed (%s): %r", msg.get(config.MSG_TYPE_KEY), e)
            return False

    async def run(self) -> None:
        backoff = config.BACKOFF_BASE_SEC
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    config.build_ws_url(),
                    additional_headers=config.WS_HEADERS,
                    subprotocols=config.WS_SUBPROTOCOLS,
                    ping_interval=config.PING_INTERVAL_SEC,
                    ping_timeout=config.PING_INTERVAL_SEC,
                    max_size=None,            # some frames can be large
                    open_timeout=config.WS_OPEN_TIMEOUT_SEC,
                ) as ws:
                    self._ws = ws
                    backoff = config.BACKOFF_BASE_SEC
                    log.info("%s connected", self.name)
                    await self._on_connect(self)
                    async for raw in ws:
                        await self._dispatch(raw)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - reconnect on any failure
                # Some websockets exceptions (e.g. InvalidURI) embed the full
                # connect URL, which carries the API key: always redact.
                log.warning("%s disconnected: %s", self.name, config.redact_url(repr(e)))
            finally:
                self._ws = None
            if self._stop.is_set():
                break
            log.info("%s reconnecting in %ss", self.name, backoff)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, config.BACKOFF_MAX_SEC)
        log.info("%s run loop stopped", self.name)

    async def _dispatch(self, raw: str | bytes) -> None:
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            log.warning("dropped non-JSON frame")
            return
        try:
            await self._on_message(data)
        except Exception:  # noqa: BLE001 - one bad message must not kill ingestion
            log.exception("message handler error (type=%s)",
                          data.get(config.MSG_TYPE_KEY) if isinstance(data, dict) else "?")

    async def close(self) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass
