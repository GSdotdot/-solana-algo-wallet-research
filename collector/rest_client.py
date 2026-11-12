"""Data-provider REST client: the one-time enrichment burst per discovered token
plus the recurring per-token stats polls.

Calls made (paths, query parameters and per-call credit costs live in
config `data_provider.requests.*`; all redacted in the public copy):
  - ohlcv            price-history backfill before discovery
  - recent_trades    recent swaps, paginated backwards in time
  - top_traders      top traders on the token
  - token_stats   holders / market cap / liquidity / windowed volume  (recurring)
  - holder_breakdown   top-10 concentration + trader-type tags             (recurring)
  - launchpad_detail bonding-curve state and reserves                    (once)
  - batch_price      batch USD price for many tokens                     (recurring)

Retry policy: 429 -> respect Retry-After; 5xx/timeout -> exponential backoff with a
bounded number of retries; other 4xx -> no retry, return None. Credits are recorded
on each 200 (estimates: the provider does not return the cost per call, so real
consumption is reconciled against its dashboard).

Key safety: the API key is sent only via config.rest_headers() and never logged;
errors log the endpoint label, never the headers.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import aiohttp

from common.params import param

from . import config
from .credits import CreditTracker
from .util import get_logger

log = get_logger("rest")

_REST = "collector.rest"
MAX_RETRIES = param(f"{_REST}.max_retries")
BACKOFF_BASE = param(f"{_REST}.backoff_base_sec")
BACKOFF_MAX = param(f"{_REST}.backoff_max_sec")
TIMEOUT_SEC = param(f"{_REST}.timeout_sec")
OHLCV_BACKFILL_SEC = param("collector.enrichment.ohlcv_backfill_sec")   # history window before discovery
RECENT_TRADES_PAGES = param("collector.enrichment.recent_trades_pages")
RECENT_TRADES_PAGE_SIZE = param("collector.enrichment.recent_trades_page_size")


def _request(name: str) -> dict[str, Any]:
    """One provider call spec: {"path", "params" (template), "credits"}."""
    return param(f"data_provider.requests.{name}")


class RestClient:
    def __init__(self, credits: CreditTracker) -> None:
        self._credits = credits
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SEC),
                headers=config.rest_headers(),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _call(self, name: str, credits: float | None = None, **values: Any) -> Any | None:
        """GET one named provider call. `values` fill the query-parameter template;
        `credits` overrides the flat per-call cost (used by the dynamic batch price)."""
        spec = _request(name)
        params = config.fill(spec["params"], **values)
        cost = spec["credits"] if credits is None else credits
        return await self._get(spec["path"], params, name, cost)

    async def _get(self, path: str, params: dict[str, Any], credit_label: str, credit_est: float) -> Any | None:
        session = await self._ensure_session()
        url = config.rest_base_url() + path
        attempt = 0
        backoff = BACKOFF_BASE
        while True:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        self._credits.add(credit_label, credit_est)
                        return await resp.json()
                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", backoff))
                        if attempt >= MAX_RETRIES:
                            log.warning("429 exhausted retries: %s", credit_label)
                            return None
                        log.warning("429 on %s; waiting %.1fs", credit_label, retry_after)
                        await asyncio.sleep(retry_after)
                    elif 500 <= resp.status < 600:
                        if attempt >= MAX_RETRIES:
                            log.warning("%d exhausted retries: %s", resp.status, credit_label)
                            return None
                        log.warning("%d on %s; backoff %.1fs", resp.status, credit_label, backoff)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, BACKOFF_MAX)
                    else:  # other 4xx -> no retry
                        body = (await resp.text())[:300]
                        log.warning("HTTP %d on %s (no retry): %s", resp.status, credit_label, body)
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt >= MAX_RETRIES:
                    log.warning("network error exhausted retries on %s: %r", credit_label, e)
                    return None
                log.warning("network error on %s: %r; backoff %.1fs", credit_label, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
            attempt += 1

    # -- recurring per-token stats polls -------------------------------------
    async def token_stats(self, token: str) -> dict | None:
        return await self._call("token_stats", token=token)

    async def holder_breakdown(self, token: str) -> dict | None:
        return await self._call("holder_breakdown", token=token)

    async def launchpad_detail(self, token: str) -> dict | None:
        return await self._call("launchpad_detail", token=token)

    async def batch_price(self, tokens: list[str]) -> dict | None:
        """Batch USD price for many tokens: the shared per-token price source.

        Chunks to BATCH_PRICE_MAX per call. The provider prices a batch sub-linearly
        in its size: credits per chunk = ceil(N ** exponent * base), recorded via the
        shared tracker. Returns {token: entry} merged across chunks
        (stats_parser.from_batch_price reads the price field), or None if every chunk
        failed (caller skips the cycle; price carries forward)."""
        if not tokens:
            return None
        spec = _request("batch_price")
        envelope = param("data_provider.response_fields.envelope")
        merged: dict[str, Any] = {}
        ok = False
        for i in range(0, len(tokens), config.BATCH_PRICE_MAX):
            chunk = tokens[i:i + config.BATCH_PRICE_MAX]
            credits = float(math.ceil((len(chunk) ** spec["credits_exponent"]) * spec["credits_base"]))
            resp = await self._call("batch_price", credits=credits, tokens=",".join(chunk))
            if resp is None:
                continue
            ok = True
            data = resp.get(envelope)
            if isinstance(data, dict):
                merged.update(data)
        return {envelope: merged} if ok else None

    async def burst(self, token: str, now: int) -> dict[str, str | None]:
        """One-time enrichment for a freshly discovered token. Returns JSON
        strings ready to store in token_identity (None where a call failed)."""
        ohlcv = await self._call("price_history", token=token,
                                 time_from=now - OHLCV_BACKFILL_SEC, time_to=now)

        trades: list[Any] = []
        before = now
        for _ in range(RECENT_TRADES_PAGES):
            page = await self._call("recent_trades", token=token, before=before, limit=RECENT_TRADES_PAGE_SIZE)
            items = _extract_items(page)
            if not items:
                break
            trades.extend(items)
            last_ts = _min_block_time(items)
            if last_ts is None or last_ts >= before:
                break
            before = last_ts

        top = await self._call("top_traders", token=token)
        # An exit-liquidity enrichment was planned but is not available for this chain,
        # so that column stays NULL.

        return {
            "ohlcv_backfill_json": _dumps(ohlcv),
            "recent_trades_json": _dumps(trades) if trades else None,
            "top_traders_json": _dumps(top),
            "exit_liquidity_json": None,
        }


def _dumps(v: Any) -> str | None:
    return None if v is None else json.dumps(v, separators=(",", ":"))


def _extract_items(page: Any) -> list[Any]:
    if not isinstance(page, dict):
        return []
    fields = param("data_provider.response_fields")
    data = page.get(fields["envelope"])
    if isinstance(data, dict):
        for key in fields["recent_trades"]["list_keys"]:
            if isinstance(data.get(key), list):
                return data[key]
        return []
    return data if isinstance(data, list) else []


def _min_block_time(items: list[Any]) -> int | None:
    key = param("data_provider.response_fields.recent_trades.block_time")
    times = [it.get(key) for it in items
             if isinstance(it, dict) and it.get(key) is not None]
    return min(times) if times else None
