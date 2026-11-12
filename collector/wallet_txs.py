"""Wallet-transaction frame parsing + trade ingestion (Dataset 1 + discovery trigger).

parse_wallet_tx() is pure and unit-tested: it turns a raw payload into a
wallet_trades row, resolving BUY / SELL / TOKEN_SWAP via the from/to stablecoin
test. Getting direction wrong silently corrupts the dataset, so it is the single
most safety-critical function here.

PAYLOAD SHAPE (verified against live frames, not the provider's documentation,
which described a different shape): each swap carries a `from` side (token spent)
and a `to` side (token received), plus the pool address, the slot, and the DEX
program address (alongside a human-readable source name). The provider's key for
each of those lives in config (`data_provider.response_fields.wallet_tx`).

TradeProcessor.handle() wires one trade through the system, in order:
  1. parse  2. detect first-touch (before subscribing)  3. position-derive
  4. write wallet_trades  5. discover/subscribe the token on first touch
  6. update lifecycle (holders/grace)  7. emit a labeled_event if it carries a label

Dataset 1 is "every transaction, no filtering": non-swaps are stored too, with
direction='OTHER' and no entry/exit semantics. Only swaps drive discovery,
position tracking, and labeling.

Known limitation: price_per_token is SOL-per-token and therefore None for
USDC-quoted swaps. PumpFun memecoin trades are SOL-quoted, so this is rare;
volume_usd is always correct regardless.
"""

from __future__ import annotations

import json
from typing import Any

from common.params import param

from . import config
from .util import get_logger

log = get_logger("wallet_txs")

_POSITION_DIRECTIONS = ("BUY", "SELL")


def _fields() -> dict[str, Any]:
    return param("data_provider.response_fields.wallet_tx")


def parse_direction(frm: dict[str, Any], to: dict[str, Any]) -> dict[str, Any]:
    """`frm` = token spent (negative side), `to` = token received (positive side).

    `quote` records which stablecoin settled the trade ('SOL' or 'USDC'). amount_sol
    is the SOL leg ONLY when SOL-quoted, else None: USDC amounts must NOT be stored
    as SOL (that conflation inflated a USDC trader's SOL PnL by the SOL price). The
    USD value of every trade is volume_usd (set in parse_wallet_tx), which is correct
    regardless of quote currency."""
    F = _fields()
    k_addr, k_amt, k_sym = F["leg_address"], F["leg_ui_amount"], F["leg_symbol"]
    f_addr, t_addr = frm.get(k_addr), to.get(k_addr)
    if f_addr in config.STABLECOINS:
        # spent SOL/USDC (from) to receive a token (to) -> BUY
        token_amt, stable_amt = to.get(k_amt), frm.get(k_amt)
        quote = "SOL" if f_addr == config.SOL_ADDRESS else "USDC"
        sol_amt = stable_amt if quote == "SOL" else None
        return {
            "direction": "BUY", "quote": quote,
            "token_address": t_addr, "token_symbol": to.get(k_sym),
            "amount_token": token_amt, "amount_sol": sol_amt,
            # SOL-per-token (None for USDC or missing amounts; never a fake 0).
            "price_per_token": (sol_amt / token_amt) if (token_amt and sol_amt is not None) else None,
        }
    if t_addr in config.STABLECOINS:
        # sold a token (from) to receive SOL/USDC (to) -> SELL
        token_amt, stable_amt = frm.get(k_amt), to.get(k_amt)
        quote = "SOL" if t_addr == config.SOL_ADDRESS else "USDC"
        sol_amt = stable_amt if quote == "SOL" else None
        return {
            "direction": "SELL", "quote": quote,
            "token_address": f_addr, "token_symbol": frm.get(k_sym),
            "amount_token": token_amt, "amount_sol": sol_amt,
            "price_per_token": (sol_amt / token_amt) if (token_amt and sol_amt is not None) else None,
        }
    # neither side is SOL/USDC -> token-to-token, log for manual review
    return {
        "direction": "TOKEN_SWAP", "quote": None,
        "token_address": f_addr, "token_symbol": frm.get(k_sym),
        "amount_token": frm.get(k_amt), "amount_sol": None,
        "price_per_token": None,
    }


def parse_wallet_tx(data: dict[str, Any], dex_map: dict[str, str]) -> dict[str, Any]:
    """Map a wallet-transaction payload to a wallet_trades row dict."""
    F = _fields()
    tx_type = data.get(F["tx_type"])
    txh = data.get(F["tx_hash"])
    frm = data.get(F["from_leg"]) or {}
    to = data.get(F["to_leg"]) or {}
    program = data.get(F["program_id"])

    row: dict[str, Any] = {
        "trade_id": txh,
        "tx_signature": txh,
        "wallet_address": data.get(F["owner"]),
        "timestamp": data.get(F["block_time"]),
        "slot_number": data.get(F["slot"]),
        "pool_address": data.get(F["pool_address"]),
        # The program id is the DEX program address; map it to a name, else fall
        # back to the provider's human-readable source name.
        "dex_program": dex_map.get(program) or data.get(F["source_name"]),
        "tx_type": tx_type,
        "volume_usd": data.get(F["volume_usd"]),
        "raw_payload": json.dumps(data, separators=(",", ":")),
    }
    if tx_type == F["swap_type_value"]:
        row.update(parse_direction(frm, to))
        row["tx_type"] = "swap"
    else:
        # Stored (no filtering) but not a trade: best-effort token, no semantics.
        row.update({
            "direction": "OTHER",
            "token_address": to.get(F["leg_address"]) or frm.get(F["leg_address"]),
            "token_symbol": to.get(F["leg_symbol"]) or frm.get(F["leg_symbol"]),
        })
    return row


class TradeProcessor:
    def __init__(self, db, subs, positions, labeler, archive, dex_map: dict[str, str], poller,
                 fees: dict[str, dict[str, float]] | None = None):
        self._db = db
        self._subs = subs
        self._positions = positions
        self._labeler = labeler
        self._archive = archive
        self._dex_map = dex_map
        self._poller = poller
        self._fees = fees or {}      # {wallet: {buy_usd, sell_usd}} priority+bribe

    async def handle(self, data: dict[str, Any]) -> None:
        row = parse_wallet_tx(data, self._dex_map)
        token = row.get("token_address")
        ts = row.get("timestamp")
        wallet = row.get("wallet_address")
        direction = row["direction"]
        is_swap = row["tx_type"] == "swap"

        if not row.get("trade_id") or ts is None or not wallet:
            log.warning("dropping malformed wallet tx: %s", row.get("trade_id"))
            return

        was_active = bool(token) and self._subs.is_active(token)
        first_touch = is_swap and direction != "OTHER" and bool(token) and not was_active

        # has_pre_context: token was already being monitored before this trade
        row["has_pre_context"] = int(was_active)
        sub_at = self._subs.subscribed_at(token) if token else None
        pre_dur = max(0, ts - sub_at) if (was_active and sub_at) else 0
        row["pre_context_duration_sec"] = pre_dur
        row["snapshot_ref_id"] = self._archive.last_snapshot_id.get(token) if token else None

        # position derivation + label (only true BUY/SELL move a position)
        label: str | None = None
        if is_swap and direction in _POSITION_DIRECTIONS:
            quote = row.get("quote") or "SOL"
            fee = config.leg_fee_usd(self._fees, wallet, direction)
            derived = self._positions.on_trade(
                wallet, token, direction,
                row.get("amount_token") or 0.0, row.get("amount_sol"),
                row.get("volume_usd") or 0.0, ts, quote=quote, leg_fee_usd=fee)
            label = derived.pop("label")
            row.update(derived)
        elif is_swap and direction == "TOKEN_SWAP":
            log.warning("TOKEN_SWAP on %s tx=%s (logged, not position-tracked)", token, row["trade_id"])

        await self._db.insert("wallet_trades", row)

        if first_touch:
            await self._subs.on_first_touch(token, wallet, ts)
        if is_swap and direction != "OTHER" and token:
            await self._subs.note_trade(token, wallet, ts)
            await self._poller.on_trade(token)   # at-trade stats poll (both states, debounced)

        if label is not None:
            await self._labeler.on_trade(row, label, token, ts, bool(was_active), pre_dur)
