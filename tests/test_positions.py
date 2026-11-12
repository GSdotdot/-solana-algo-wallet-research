"""Position tracking and trade labelling (collector/positions.py).

Offline: no network, no database. Wallet / token names and all amounts are synthetic.
"""

from collector.positions import DUST_FRACTION, FLIP_WINDOW_SEC, PositionBook

E = 1e-6
BUY_FEE, SELL_FEE = 0.05, 0.03          # made-up per-leg priority+bribe costs, USD


def test_sol_round_trip_with_fees():
    pb = PositionBook()
    b = pb.on_trade("W", "Ta", "BUY", 1000, 1.0, 64.0, 0, quote="SOL", leg_fee_usd=BUY_FEE)
    assert b["is_entry"] == 1 and b["position_pnl_usd"] is None and b["position_pnl_sol"] is None
    assert b["prio_bribe_fee_usd"] == BUY_FEE
    s = pb.on_trade("W", "Ta", "SELL", 1000, 2.0, 128.0, 10, quote="SOL", leg_fee_usd=SELL_FEE)
    assert s["is_exit"] == 1 and s["hold_duration_sec"] == 10
    assert abs(s["position_pnl_usd"] - 64.0) < E
    assert abs(s["net_pnl_usd"] - (64.0 - BUY_FEE - SELL_FEE)) < E
    assert abs(s["position_pnl_sol"] - 1.0) < E
    assert pb.holders("Ta") == set()


def test_usdc_round_trip_never_reports_usdc_as_sol():
    pb = PositionBook()
    pb.on_trade("W", "Tb", "BUY", 1000, None, 50.0, 0, quote="USDC", leg_fee_usd=BUY_FEE)
    s = pb.on_trade("W", "Tb", "SELL", 1000, None, 80.0, 5, quote="USDC", leg_fee_usd=SELL_FEE)
    assert abs(s["position_pnl_usd"] - 30.0) < E
    assert s["position_pnl_sol"] is None
    assert abs(s["net_pnl_usd"] - (30.0 - BUY_FEE - SELL_FEE)) < E


def test_oversized_sell_is_prorated_to_tracked_size():
    """Selling 10x the tracked size must only realise PnL on the tracked portion."""
    pb = PositionBook()
    pb.on_trade("W", "Tc", "BUY", 100, 1.0, 64.0, 0, quote="SOL")
    s = pb.on_trade("W", "Tc", "SELL", 1000, 20.0, 1280.0, 3, quote="SOL")
    assert abs(s["position_pnl_usd"] - 64.0) < E          # not 1280 - 64
    assert abs(s["position_pnl_sol"] - 1.0) < E


def test_untracked_sell_has_unknown_pnl_not_zero():
    """An exit whose entry predates coverage: PnL is None, never fabricated."""
    pb = PositionBook()
    s = pb.on_trade("W", "Td", "SELL", 500, 5.0, 320.0, 0, quote="SOL")
    assert s["is_exit"] == 1 and s["position_pnl_usd"] is None and s["net_pnl_usd"] is None


def test_partial_then_full_exit_attributes_buy_fee_pro_rata():
    pb = PositionBook()
    pb.on_trade("W", "Te", "BUY", 1000, 1.0, 64.0, 0, quote="SOL", leg_fee_usd=BUY_FEE)
    p1 = pb.on_trade("W", "Te", "SELL", 500, 1.0, 64.0, 5, quote="SOL", leg_fee_usd=SELL_FEE)
    assert p1["is_partial_exit"] == 1 and abs(p1["position_pnl_usd"] - 32.0) < E
    assert abs(p1["net_pnl_usd"] - (32.0 - BUY_FEE / 2 - SELL_FEE)) < E
    p2 = pb.on_trade("W", "Te", "SELL", 500, 1.5, 96.0, 9, quote="SOL", leg_fee_usd=SELL_FEE)
    assert p2["is_exit"] == 1 and abs(p2["position_pnl_usd"] - 64.0) < E
    total_fees = (32.0 + 64.0) - (p1["net_pnl_usd"] + p2["net_pnl_usd"])
    assert abs(total_fees - (BUY_FEE + 2 * SELL_FEE)) < E


def test_dust_remainder_counts_as_full_exit():
    pb = PositionBook()
    pb.on_trade("W", "Tf", "BUY", 1000, 1.0, 64.0, 0)
    leave = 1000 * DUST_FRACTION / 2                      # below the dust threshold
    s = pb.on_trade("W", "Tf", "SELL", 1000 - leave, 1.0, 64.0, 4)
    assert s["is_exit"] == 1 and s["is_partial_exit"] == 0


def test_quick_rebuy_is_flip_slow_rebuy_is_entry():
    pb = PositionBook()
    pb.on_trade("W", "Tg", "BUY", 100, 1.0, 64.0, 0)
    pb.on_trade("W", "Tg", "SELL", 100, 1.0, 64.0, 10)
    flip = pb.on_trade("W", "Tg", "BUY", 100, 1.0, 64.0, 10 + FLIP_WINDOW_SEC)
    assert flip["is_flip"] == 1 and flip["is_entry"] == 0
    pb.on_trade("W", "Tg", "SELL", 100, 1.0, 64.0, 100)
    late = pb.on_trade("W", "Tg", "BUY", 100, 1.0, 64.0, 100 + FLIP_WINDOW_SEC + 1)
    assert late["is_entry"] == 1 and late["is_flip"] == 0


def test_add_and_other_wallets_active():
    pb = PositionBook()
    pb.on_trade("W1", "Th", "BUY", 100, 1.0, 64.0, 0)
    add = pb.on_trade("W1", "Th", "BUY", 50, 0.5, 32.0, 2)
    assert add["is_add"] == 1
    other = pb.on_trade("W2", "Th", "BUY", 10, 0.1, 6.4, 3)
    assert other["other_wallets_active"] == 1
    assert pb.holders("Th") == {"W1", "W2"}
