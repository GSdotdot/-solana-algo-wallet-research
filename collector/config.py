"""Configuration, secrets, and runtime tunables.

Hard rule: the data-provider API key is read ONLY via get_api_key() from the
environment (loaded from .env). It is never printed, logged, or interpolated into
a logged string. build_ws_url() returns the live URL with the key filled in;
redact_url() is the only thing safe to log.

Every tunable (cadences, lifecycle timeouts, retention, budget guards) and every
data-provider detail (URLs, message types, endpoint paths, field names) is loaded
through common.params from a YAML file. In this public copy they are placeholders
(see config.example.yaml); an environment variable, where one is listed, still
overrides the file so a bounded test run never needs an edit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from common.params import is_redacted, param

from .schema import WINDOWS

# --- paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
# WRE_DB_PATH overrides the collector DB (e.g. a dedicated smoke DB so a test run
# never appends to the canonical store). Default = data/wallet_research.db.
DB_PATH = Path(os.environ.get("WRE_DB_PATH") or (DATA_DIR / "wallet_research.db"))
WALLETS_PATH = CONFIG_DIR / "wallets.json"
WALLET_FEES_PATH = CONFIG_DIR / "wallet_fees.json"
DEX_PROGRAMS_PATH = CONFIG_DIR / "dex_programs.json"

load_dotenv(PROJECT_ROOT / ".env")  # harmless if absent

API_KEY_ENV = "DATA_PROVIDER_API_KEY"
REST_URL_ENV = "DATA_PROVIDER_REST_URL"
WS_URL_ENV = "DATA_PROVIDER_WS_URL"


def _env_num(env: str, key: str, cast=int) -> Any:
    """Environment override if set, else the YAML value (a Redacted placeholder in
    the public copy, which fails loudly only when it is actually used)."""
    raw = os.environ.get(env)
    return cast(raw) if raw else param(key)


def _env_flag(env: str, default: bool = True) -> bool:
    raw = os.environ.get(env)
    return default if raw is None else raw not in ("0", "false", "")


def _labels(value: Any) -> frozenset[str]:
    """A list of wallet labels from YAML -> frozenset. Redacted / empty -> empty set."""
    if value is None or is_redacted(value):
        return frozenset()
    return frozenset(str(s).strip() for s in value if not is_redacted(s) and str(s).strip())


# --- data-provider transport ------------------------------------------------
_WS = "data_provider.ws"
WS_URL_TEMPLATE = param(f"{_WS}.url_template")        # formatted with {base} and {key}
WS_HEADERS = param(f"{_WS}.headers")                  # extra handshake headers, if any
# The provider requires a named subprotocol. It MUST be offered via the websockets
# `subprotocols=` param, not a raw header: otherwise the library raises a
# NegotiationError when the server echoes a selection it never saw offered.
WS_SUBPROTOCOLS = param(f"{_WS}.subprotocols")
REST_HEADERS_BASE = param("data_provider.rest.base_headers")
REST_AUTH_HEADER = param("data_provider.rest.auth_header")

# WebSocket frame vocabulary (provider-specific, redacted).
MSG_TYPE_KEY = param(f"{_WS}.type_key")               # frame field holding the event type
MSG_DATA_KEY = param(f"{_WS}.data_key")               # frame field holding the payload
MSG_WALLET_TXS = param(f"{_WS}.wallet_txs_event")     # inbound: one tracked-wallet transaction
MSG_PRICE = param(f"{_WS}.price_event")               # inbound: one 1s price bar

# --- WebSocket behaviour ----------------------------------------------------
PING_INTERVAL_SEC = param("collector.ws.ping_interval_sec")
BACKOFF_BASE_SEC = param("collector.ws.backoff_base_sec")
BACKOFF_MAX_SEC = param("collector.ws.backoff_max_sec")
WS_OPEN_TIMEOUT_SEC = param("collector.ws.open_timeout_sec")
# Granularity of the per-token price stream opened for the fast-scalper ("B") wallets.
# The provider's price channel is single-active per connection, so each B-held token
# gets its own socket (collector.price_sockets).
PRICE_CHART_TYPE = param(f"{_WS}.price_bar_granularity")
# Rolling-window suffixes the provider can emit; the assert guarantees our schema
# never expects a window the provider can't produce.
AVAILABLE_INTERVALS = ("1m", "5m", "30m", "1h", "2h", "4h", "8h", "24h")
assert set(WINDOWS) <= set(AVAILABLE_INTERVALS), (
    "schema.WINDOWS expects intervals the provider cannot emit", set(WINDOWS) - set(AVAILABLE_INTERVALS))

# --- archive / write buffering ---------------------------------------------
SNAPSHOT_INTERVAL_SEC = param("collector.archive.snapshot_interval_sec")
BUFFER_SIZE = param("collector.archive.buffer_size")          # flush snapshot buffer at N rows ...
FLUSH_INTERVAL_SEC = param("collector.archive.flush_interval_sec")  # ... or after this long, whichever first

# --- per-token stats polling (REST) -----------------------------------------
# Base poll cadence = token-stats cadence (holder/mcap/liquidity/volume).
STATS_POLL_INTERVAL_SEC = _env_num("WRE_STATS_POLL_SEC", "collector.polling.stats_poll_interval_sec")

# --- batch price poll (REST) ------------------------------------------------
# Fills `price` for ALL active tokens on a short cadence. Runs subject to the hard
# ceiling only, NOT the State-2 throttle (price is the priority signal).
PRICE_POLL_INTERVAL_SEC = _env_num("WRE_PRICE_POLL_SEC", "collector.polling.price_poll_interval_sec")
BATCH_PRICE_MAX = _env_num("WRE_BATCH_PRICE_MAX", "collector.polling.batch_price_max")   # tokens per batch call

# --- stats-poll cadence tiers (per token, by lifecycle phase) ---------------
# Replaces a flat "poll every active token the same" cadence: a token costs the
# most while a wallet actually holds it, and very little once everyone has exited.
#   full   = currently held, OR within POST_EXIT_FULL_SEC of the last holder exit
#            -> token stats @ STATS_POLL_INTERVAL_SEC, holder breakdown
#               @ HOLDER_BREAKDOWN_FULL_SEC, price at the batch cadence (or the
#               dedicated stream for B-wallet tokens).
#   coarse = in grace, > POST_EXIT_FULL_SEC since the last trade
#            -> token stats @ STATS_POLL_COARSE_SEC, NO background holder breakdown.
# The batch price call covers every active token in both tiers (except tokens served by a
# dedicated targeted-B price socket).
# The at-trade poll (token-stats + holder breakdown on every entry/exit) is deliberately
# UNCHANGED in both tiers: that is the decision-moment snapshot.
POST_EXIT_FULL_SEC = _env_num("WRE_POST_EXIT_FULL_SEC", "collector.polling.post_exit_full_sec")
STATS_POLL_COARSE_SEC = _env_num("WRE_STATS_POLL_COARSE_SEC", "collector.polling.stats_poll_coarse_sec")
HOLDER_BREAKDOWN_FULL_SEC = _env_num("WRE_HOLDER_BREAKDOWN_FULL_SEC", "collector.polling.holder_breakdown_full_sec")

# --- adaptive throttle (pace guard) -----------------------------------------
# Emergency API-credit brake layered ON TOP of the cadence tiers: switch to State 2
# (sticky) once cumulative REST credits exceed CREDIT_PACE_PER_DAY * (elapsed / 1 day),
# evaluated after THROTTLE_WARMUP_SEC so discovery bursts / cold start don't trip
# it. In State 2 the full-tier token-stats cadence is demoted to the coarse cadence.
CREDIT_PACE_PER_DAY = _env_num("WRE_CREDIT_PACE_PER_DAY", "collector.budget.pace_per_day", float)
THROTTLE_WARMUP_SEC = _env_num("WRE_THROTTLE_WARMUP_SEC", "collector.budget.throttle_warmup_sec")

# Hard backstop: if cumulative REST credits reach this, STOP all REST polling
# (background + at-trade); WS ingestion continues. Guarantees we never blow past.
CREDIT_HARD_CEILING = _env_num("WRE_CREDIT_HARD_CEILING", "collector.budget.hard_ceiling", float)

# At-trade polling (BOTH states): poll a token immediately on a wallet trade to
# capture the exact decision-moment context (a trade can land up to a full
# background interval after the last poll). Debounced against the background poll.
AT_TRADE_POLL = _env_flag("WRE_AT_TRADE_POLL")
AT_TRADE_DEBOUNCE_SEC = _env_num("WRE_AT_TRADE_DEBOUNCE_SEC", "collector.polling.at_trade_debounce_sec")

# --- targeted-B per-token price sockets -------------------------------------
# A subset of the tracked wallets trade on holds too short for the shared batch
# price to resolve. Each token such a "B" wallet holds gets its own price socket
# that feeds a fine-grained `price` + the price_ticks archive, and is dropped from
# the batch price call while the socket is open. Which wallets are in the subset is
# a research finding and lives in the private config; WRE_B_WALLETS="LabelA,LabelB"
# overrides it (matched on the wallets.json `label`). Empty disables targeted-B
# entirely (everything falls back to the batch price).
_b_env = os.environ.get("WRE_B_WALLETS")
B_WALLET_LABELS = (frozenset(s.strip() for s in _b_env.split(",") if s.strip())
                   if _b_env is not None else _labels(param("collector.b_wallets.labels")))
# Keep a B token's dedicated price socket open through the hold and this long after
# the last B-wallet exits (re-entry cancels the timer); then close it and fall back
# to the batch price if a non-B wallet still holds it.
B_PRICE_POST_EXIT_SEC = _env_num("WRE_B_PRICE_POST_EXIT_SEC", "collector.b_wallets.price_post_exit_sec")

# --- raw price-bar archive (persists the targeted-B price stream) -----------
# Rolling-window pruned so it stays bounded; retention is much longer than the
# periodic export interval so nothing is lost before pruning (collector.tick_archive).
STORE_TICKS = _env_flag("WRE_STORE_TICKS")
TICK_RETENTION_HRS = _env_num("WRE_TICK_RETENTION_HRS", "collector.ticks.retention_hrs")   # rolling window
TICK_FLUSH_INTERVAL_SEC = _env_num("WRE_TICK_FLUSH_SEC", "collector.ticks.flush_interval_sec")
TICK_PRUNE_INTERVAL_SEC = _env_num("WRE_TICK_PRUNE_SEC", "collector.ticks.prune_interval_sec")

# --- subscription lifecycle -------------------------------------------------
# Force-unsubscribe after this long with no tracked-wallet trade on the token.
HARD_CEILING_SEC = param("collector.lifecycle.hard_ceiling_sec")

# Grace tiers: when the last tracked wallet exits, hold the subscription this long
# in case of re-entry. Tier label -> seconds.
GRACE_TIERS = param("collector.lifecycle.grace_tiers")          # {"short": s, "medium": s, "long": s}

# A hold shorter than this marks a "quick trade" (one-and-done scalp).
QUICK_HOLD_SEC = param("collector.lifecycle.quick_hold_sec")
# This many trades on the token today marks it "frequently re-entered".
FREQUENT_TRADES_TODAY = param("collector.lifecycle.frequent_trades_today")


def pick_grace_tier(
    trades_today: int,
    last_hold_duration_sec: int | None = None,
) -> tuple[str, int]:
    """Grace tier from tracked-wallet activity on this token *today* (UTC).

      - traded FREQUENT_TRADES_TODAY+ times today  -> long   (high re-entry: sell-rebuy)
      - traded fewer times today                   -> medium (moderate re-entry)
      - single quick trade (hold < QUICK_HOLD_SEC) -> short  (likely one-and-done)

    The frequent check wins first; below that, a quick hold downgrades to short,
    otherwise the token gets medium. `last_hold_duration_sec` is the hold of the
    position that just closed (None if unknown -> treated as not-quick).
    """
    if trades_today >= FREQUENT_TRADES_TODAY:
        return "long", GRACE_TIERS["long"]
    if last_hold_duration_sec is not None and last_hold_duration_sec < QUICK_HOLD_SEC:
        return "short", GRACE_TIERS["short"]
    return "medium", GRACE_TIERS["medium"]


# --- trade direction parsing ------------------------------------------------
# Public chain constants (the wrapped-SOL and USDC mints), not project-specific.
SOL_ADDRESS = "So11111111111111111111111111111111111111112"   # WSOL is the same mint
USDC_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
STABLECOINS = frozenset({SOL_ADDRESS, USDC_ADDRESS})


# --- secrets ----------------------------------------------------------------
def get_api_key() -> str:
    """The provider key. Never log the return value or anything derived from it."""
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is not set. Put it in .env (never commit it).")
    return key


def rest_base_url() -> str:
    url = os.environ.get(REST_URL_ENV)
    if not url:
        raise RuntimeError(f"{REST_URL_ENV} is not set. Put it in .env.")
    return url.rstrip("/")


def build_ws_url() -> str:
    """Live WS URL with the key filled in. Do NOT log this."""
    base = os.environ.get(WS_URL_ENV)
    if not base:
        raise RuntimeError(f"{WS_URL_ENV} is not set. Put it in .env.")
    return WS_URL_TEMPLATE.format(base=base.rstrip("/"), key=get_api_key())


def rest_headers() -> dict[str, str]:
    return {**REST_HEADERS_BASE, REST_AUTH_HEADER: get_api_key()}


def redact_url(url: str) -> str:
    """Safe-to-log form of any URL that may carry the key."""
    key = os.environ.get(API_KEY_ENV)
    return url.replace(key, "***REDACTED***") if key else url


# --- provider templates -----------------------------------------------------
def fill(template: Any, **values: Any) -> Any:
    """Fill {placeholders} in a provider message / request template from YAML.
    Works through nested dicts and lists; non-string leaves pass through. A string
    that is exactly one placeholder keeps the value's type (ints stay ints)."""
    if isinstance(template, dict):
        return {k: fill(v, **values) for k, v in template.items()}
    if isinstance(template, list):
        return [fill(v, **values) for v in template]
    if isinstance(template, str):
        stripped = template.strip()
        if stripped.startswith("{") and stripped.endswith("}") and stripped[1:-1] in values:
            return values[stripped[1:-1]]
        return template.format(**values)
    return template


def sub_wallet(wallet: str) -> dict:
    """Subscribe frame for one wallet's transaction stream (single-active per
    connection, so one socket carries one wallet)."""
    return fill(param(f"{_WS}.subscribe_wallet_template"), wallet=wallet)


def sub_price(token: str) -> dict:
    """Subscribe frame for one token's price stream (the only known-good shape,
    verified live). Single-active per connection, so one socket carries one token."""
    return fill(param(f"{_WS}.subscribe_price_template"), token=token, granularity=PRICE_CHART_TYPE)


# --- config files -----------------------------------------------------------
def load_wallets() -> list[dict[str, str]]:
    """Read config/wallets.json -> list of {address, label}. Runtime-updatable.

    The real watchlist is private and gitignored; config/wallets.example.json shows
    the shape with pseudonyms."""
    if not WALLETS_PATH.exists():
        raise RuntimeError(
            f"{WALLETS_PATH.name} not found. Copy wallets.example.json and fill in the "
            "real addresses (this file is gitignored).")
    data = json.loads(WALLETS_PATH.read_text(encoding="utf-8"))
    wallets = data.get("target_wallets") or []
    cleaned = [w for w in wallets if w.get("address") and not str(w["address"]).startswith("<")]
    if not cleaned:
        raise RuntimeError(f"{WALLETS_PATH.name} has no real wallet addresses (only placeholders?).")
    # Smoke-run knob: WRE_MAX_WALLETS=1 subscribes to only the first N wallets
    # WITHOUT editing wallets.json. Unset in normal operation -> all wallets.
    limit = os.environ.get("WRE_MAX_WALLETS")
    if limit:
        cleaned = cleaned[: int(limit)]
    return cleaned


def b_wallet_addresses(wallets: list[dict[str, str]]) -> set[str]:
    """Resolve B_WALLET_LABELS -> addresses against the loaded wallet list (matched
    on `label`). Empty if none match (e.g. WRE_MAX_WALLETS trimmed them, or a label
    typo); the caller logs the resolved count so a silent zero-match is visible."""
    return {w["address"] for w in wallets
            if w.get("address") and w.get("label") in B_WALLET_LABELS}


def load_dex_programs() -> dict[str, str]:
    """Router/program address -> DEX name. Empty if the file is missing."""
    if not DEX_PROGRAMS_PATH.exists():
        return {}
    data = json.loads(DEX_PROGRAMS_PATH.read_text(encoding="utf-8"))
    return data.get("programs") or {}


# Default per-leg priority+bribe (tip) cost in USD when a wallet has no entry
# in wallet_fees.json. 0 = unknown -> net_pnl_usd == gross until fees are supplied.
_DEFAULT_FEE = {"buy_usd": 0.0, "sell_usd": 0.0}


def load_wallet_fees() -> dict[str, dict[str, float]]:
    """Per-wallet priority+bribe fee (USD per leg), address-keyed.

    config/wallet_fees.json (gitignored, like wallets.json):
      {"default": {"buy_usd": 0.0, "sell_usd": 0.0},
       "wallets": {"<addr>": {"buy_usd": ..., "sell_usd": ...}, ...}}

    Returns {address: {"buy_usd", "sell_usd"}} merged over the default. Missing file
    -> empty (every lookup falls back to the zero default, so PnL stays gross)."""
    if not WALLET_FEES_PATH.exists():
        return {}
    data = json.loads(WALLET_FEES_PATH.read_text(encoding="utf-8"))
    default = {**_DEFAULT_FEE, **(data.get("default") or {})}
    out: dict[str, dict[str, float]] = {}
    for addr, fee in (data.get("wallets") or {}).items():
        if addr and not str(addr).startswith("<"):
            out[addr] = {**default, **(fee or {})}
    return out


def leg_fee_usd(fees: dict[str, dict[str, float]], wallet: str, direction: str) -> float:
    """Look up one leg's priority+bribe cost (USD) for a wallet + 'BUY'/'SELL'.
    Unknown wallet -> 0.0 (gross == net until fees are configured)."""
    f = fees.get(wallet)
    if not f:
        return 0.0
    return float(f.get("buy_usd" if direction == "BUY" else "sell_usd", 0.0) or 0.0)
