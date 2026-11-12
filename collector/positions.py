"""Per-(wallet, token) position tracking + trade labeling (Module B core).

Pure in-memory bookkeeping, zero API cost. Feeds three consumers:
  - wallet_trades derived columns (is_entry/exit/add/..., pnl, hold, sequence)
  - labeled_events label (ENTRY/EXIT/ADD/PARTIAL_EXIT/FLIP)
  - subscription grace logic (holders, trades_today, last closed hold)

State is intentionally not persisted: on restart we start fresh, so positions
opened before a restart show no tracked basis (label EXIT with None pnl/hold).
That is the documented pre-context gap, not a bug.

ASSUMPTIONS (tunable, in config):
  - DUST_FRACTION: a sell leaving no more than this fraction of the episode's peak
    tokens counts as a full EXIT (handles "sold almost everything" dust remainders)
    rather than PARTIAL_EXIT.
  - FLIP_WINDOW_SEC: a re-open within this long of a full exit is labeled FLIP, not
    ENTRY (captures the sell-rebuy pattern as its own event type).
  - Cost basis is weighted-average; realised pnl on a sell = proceeds minus the
    avg-cost basis of the tokens sold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.params import param

from .util import utc_day_start

EPS = 1e-9
DUST_FRACTION = param("collector.positions.dust_fraction")       # tuned privately
FLIP_WINDOW_SEC = param("collector.positions.flip_window_sec")   # tuned privately


@dataclass
class _Pos:
    tokens: float = 0.0
    cost_usd: float = 0.0          # USD basis (canonical, from volume_usd)
    cost_sol: float = 0.0          # SOL basis — meaningful only while sol_clean
    buy_fees_usd: float = 0.0      # accumulated priority+bribe on the open buys (deferred to sells)
    sol_clean: bool = True         # False once any USDC-quoted leg touches this position
    opened_at: int | None = None
    peak: float = 0.0


@dataclass
class PositionBook:
    _pos: dict[tuple[str, str], _Pos] = field(default_factory=dict)
    _last_any: dict[str, int] = field(default_factory=dict)
    _last_tok: dict[tuple[str, str], int] = field(default_factory=dict)
    _seq: dict[str, int] = field(default_factory=dict)
    _trades_ts: dict[str, list[int]] = field(default_factory=dict)
    _last_exit: dict[tuple[str, str], int] = field(default_factory=dict)
    _last_hold: dict[str, int] = field(default_factory=dict)

    def on_trade(
        self,
        wallet: str,
        token: str,
        direction: str,            # 'BUY' | 'SELL' (TOKEN_SWAP is not position-tracked)
        amount_token: float,
        amount_sol: float | None,  # SOL leg amount; None when the quote side is USDC
        amount_usd: float,         # volume_usd — the trade's USD value (always present)
        ts: int,
        quote: str = "SOL",        # 'SOL' | 'USDC' — which stablecoin the trade settled in
        leg_fee_usd: float = 0.0,  # this leg's priority+bribe (tip) cost, USD
    ) -> dict[str, Any]:
        key = (wallet, token)
        amount_token = amount_token or 0.0
        amount_usd = amount_usd or 0.0
        sol_amt = amount_sol if (quote == "SOL" and amount_sol is not None) else None

        tsla_any = ts - self._last_any[wallet] if wallet in self._last_any else None
        tsla_tok = ts - self._last_tok[key] if key in self._last_tok else None

        self._seq[token] = self._seq.get(token, 0) + 1
        seq = self._seq[token]
        self._trades_ts.setdefault(token, []).append(ts)

        p = self._pos.setdefault(key, _Pos())
        label: str | None = None
        pnl_usd: float | None = None       # gross realized USD (net-of-DEX), canonical
        net_usd: float | None = None       # gross minus attributed priority+bribe
        pnl_sol: float | None = None       # precise SOL, only for fully-SOL-quoted positions
        hold: int | None = None

        if direction == "BUY":
            if p.tokens <= EPS:
                le = self._last_exit.get(key)
                label = "FLIP" if (le is not None and ts - le <= FLIP_WINDOW_SEC) else "ENTRY"
                # reopen: reset basis/episode
                p.tokens = 0.0; p.cost_usd = 0.0; p.cost_sol = 0.0
                p.buy_fees_usd = 0.0; p.sol_clean = True; p.peak = 0.0
                p.opened_at = ts
            else:
                label = "ADD"
            p.tokens += amount_token
            p.cost_usd += amount_usd
            if sol_amt is not None:
                p.cost_sol += sol_amt
            else:
                p.sol_clean = False                 # USDC-quoted buy taints the SOL basis
            p.buy_fees_usd += (leg_fee_usd or 0.0)  # deferred; attributed to sells pro-rata
            p.peak = max(p.peak, p.tokens)

        elif direction == "SELL":
            if sol_amt is None:
                p.sol_clean = False                 # USDC-quoted sell -> SOL pnl not expressible
            if p.tokens <= EPS:
                # selling with no tracked basis (position opened before our coverage) ->
                # pnl is unknown (None), not zero; do not fabricate it.
                label = "EXIT"
                self._last_exit[key] = ts
                p.tokens = 0.0; p.cost_usd = 0.0; p.cost_sol = 0.0
                p.buy_fees_usd = 0.0; p.opened_at = None
            else:
                sold = min(amount_token, p.tokens)
                frac = sold / p.tokens                                  # share of position closed
                basis_usd = p.cost_usd * frac
                proceeds_usd = amount_usd * (sold / amount_token) if amount_token > 0 else 0.0
                pnl_usd = proceeds_usd - basis_usd
                attributed_buy_fee = p.buy_fees_usd * frac
                net_usd = pnl_usd - attributed_buy_fee - (leg_fee_usd or 0.0)
                if p.sol_clean and sol_amt is not None:
                    basis_sol = p.cost_sol * frac
                    proceeds_sol = sol_amt * (sold / amount_token) if amount_token > 0 else 0.0
                    pnl_sol = proceeds_sol - basis_sol
                p.tokens -= sold
                p.cost_usd -= basis_usd
                p.cost_sol -= p.cost_sol * frac
                p.buy_fees_usd -= attributed_buy_fee
                if p.tokens <= max(EPS, p.peak * DUST_FRACTION):
                    label = "EXIT"
                    if p.opened_at is not None:
                        hold = ts - p.opened_at
                        self._last_hold[token] = hold
                    self._last_exit[key] = ts
                    p.tokens = 0.0; p.cost_usd = 0.0; p.cost_sol = 0.0
                    p.buy_fees_usd = 0.0; p.opened_at = None
                else:
                    label = "PARTIAL_EXIT"
                    if p.opened_at is not None:
                        hold = ts - p.opened_at
        # TOKEN_SWAP -> label stays None (logged upstream, no position change)

        self._last_any[wallet] = ts
        self._last_tok[key] = ts

        others = sum(
            1 for (w2, t2), pp in self._pos.items()
            if t2 == token and w2 != wallet and pp.tokens > EPS
        )

        return {
            "label": label,
            "is_entry": int(label == "ENTRY"),
            "is_exit": int(label == "EXIT"),
            "is_add": int(label == "ADD"),
            "is_partial_exit": int(label == "PARTIAL_EXIT"),
            "is_flip": int(label == "FLIP"),
            # Realized PnL: USD canonical; SOL precise-or-None; net = gross - prio+bribe.
            "position_pnl_usd": pnl_usd,
            "net_pnl_usd": net_usd,
            "prio_bribe_fee_usd": (leg_fee_usd or 0.0) if direction in ("BUY", "SELL") else None,
            "position_pnl_sol": pnl_sol,
            "hold_duration_sec": hold,
            "cumulative_position_tokens": p.tokens,
            "trade_sequence_on_token": seq,
            "time_since_last_trade_any": tsla_any,
            "time_since_last_trade_same_token": tsla_tok,
            "other_wallets_active": others,
        }

    # -- queries for subscription lifecycle ----------------------------------
    def holders(self, token: str) -> set[str]:
        """Tracked wallets currently holding a non-dust position in this token."""
        return {w for (w, t), p in self._pos.items() if t == token and p.tokens > EPS}

    def trades_today(self, token: str, now: int) -> int:
        day = utc_day_start(now)
        return sum(1 for x in self._trades_ts.get(token, ()) if x >= day)

    def last_hold(self, token: str) -> int | None:
        return self._last_hold.get(token)
