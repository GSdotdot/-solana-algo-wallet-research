"""Fee accounting for net-PnL analysis (pure, no runtime/collector dependency).

KEY FINDING, verified against live data: the DEX swap fee + slippage is **already
embedded** in the recorded trade amounts. The recorded SOL amount is the wallet's
actual balance change, so the fee taken inside the swap is already reflected in
SOL-in / SOL-out. This was confirmed by comparing execution prices with a reference
price: buys land above it and sells below it, by a wedge consistent with the DEX's
published fee tiers. Therefore gross realised PnL is ALREADY net-of-DEX; do NOT
subtract the tier again (that would double-count).

The only costs NOT embedded in PnL are PRIORITY + BRIBE (validator tip): separate
network spend, not part of the swap legs. These are folded in live: per-wallet
per-leg USD costs live in config/wallet_fees.json, positions.py attributes them per
position (buy-leg fees deferred + the sell-leg fee), and wallet_trades stores
position_pnl_usd (gross), prio_bribe_fee_usd (this leg), and net_pnl_usd (net).

PnL is USD-canonical (from volume_usd, currency-agnostic: the SOL columns mixed SOL
and USDC and were unreliable for USDC traders). net_pnl_usd() / net_pnl_sol()
below are the pure reconciliation helpers.
"""
from __future__ import annotations

from common.params import param

# Nominal DEX fee by market cap: [[mcap_ceiling_usd, rate], ...] with the last tier
# as the open-ended fallback. This is the fee ALREADY embedded in realised PnL; it
# is kept for reconciliation/estimation only, NOT to re-subtract.
DEX_FEE_TIERS = param("collector.fees.dex_fee_tiers")


def dex_fee_rate(market_cap: float | None) -> float:
    """Nominal DEX fee rate for a trade at this market cap."""
    for cap, rate in DEX_FEE_TIERS[:-1]:
        if market_cap is not None and market_cap < cap:
            return rate
    return DEX_FEE_TIERS[-1][1]


def dex_fee_estimate_sol(leg_volume_sol: float, market_cap: float | None) -> float:
    """Nominal DEX fee for ONE leg (already embedded in PnL; reconciliation only)."""
    return (leg_volume_sol or 0.0) * dex_fee_rate(market_cap)


def effective_cost_frac(exec_price: float | None, reference_price: float | None) -> float | None:
    """Actual per-leg fee+slippage measured from the exec-vs-reference wedge.
    Signed: +ve = paid above reference (buy) / received below reference (sell).
    Note this nets the wallet's own execution skill, so it can be favourable for
    a fast trader (e.g. a scalper buying below / selling above the lagging ref)."""
    if not reference_price:
        return None
    return (exec_price - reference_price) / reference_price


def net_pnl_sol(gross_pnl_sol: float, priority_sol: float = 0.0, bribe_sol: float = 0.0) -> float:
    """`gross_pnl_sol` (position_pnl_sol) is ALREADY net of DEX fee + slippage;
    deduct the separate priority + bribe network spend."""
    return gross_pnl_sol - (priority_sol or 0.0) - (bribe_sol or 0.0)


def net_pnl_usd(gross_pnl_usd: float, prio_bribe_usd: float = 0.0) -> float:
    """USD realised PnL (from volume_usd) is already net-of-DEX; deduct the priority+
    bribe leg costs. `prio_bribe_usd` for a position = its attributed buy-leg fees +
    the sell-leg fee (positions.py does this attribution); this helper is the pure
    subtraction for ad-hoc reconciliation."""
    return (gross_pnl_usd or 0.0) - (prio_bribe_usd or 0.0)
