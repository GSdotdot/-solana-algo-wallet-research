"""Provider-payload parsers (collector/wallet_txs.py, stats_parser.py, fees.py, config.fill).

Payloads are synthetic and use the invented field names from tests/test_config.yaml;
tokens and wallets are placeholders.
"""

from collector import config, fees
from collector import stats_parser as sp
from collector.wallet_txs import parse_wallet_tx

DEX = {"PROGRAM_X": "Example AMM"}


def _swap(spent_mint, received_mint, spent_qty=2.0, received_qty=500_000.0):
    return {"kind": "trade", "sig": "SIG_1", "wallet": "WALLET_TEST_A", "ts": 1_700_000_000,
            "slot": 123, "pool": "POOL_1", "program": "PROGRAM_X", "venue": "example_amm", "usd": 300.0,
            "spent": {"mint": spent_mint, "qty": spent_qty, "sym": "S"},
            "received": {"mint": received_mint, "qty": received_qty, "sym": "R"}}


def test_buy_sell_direction_and_price():
    buy = parse_wallet_tx(_swap(config.SOL_ADDRESS, "TOKEN_A"), DEX)
    assert buy["direction"] == "BUY" and buy["quote"] == "SOL" and buy["token_address"] == "TOKEN_A"
    assert buy["amount_sol"] == 2.0 and buy["amount_token"] == 500_000.0
    assert abs(buy["price_per_token"] - 2.0 / 500_000.0) < 1e-15
    assert buy["dex_program"] == "Example AMM"            # mapped from the program id
    assert buy["pool_address"] == "POOL_1" and buy["slot_number"] == 123 and buy["volume_usd"] == 300.0

    sell = parse_wallet_tx(_swap("TOKEN_A", config.SOL_ADDRESS, 500_000.0, 2.0), DEX)
    assert sell["direction"] == "SELL" and sell["quote"] == "SOL" and sell["token_address"] == "TOKEN_A"
    assert sell["amount_sol"] == 2.0


def test_usdc_quote_is_never_stored_as_sol():
    ub = parse_wallet_tx(_swap(config.USDC_ADDRESS, "TOKEN_A", 300.0), DEX)
    assert ub["direction"] == "BUY" and ub["quote"] == "USDC"
    assert ub["amount_sol"] is None and ub["price_per_token"] is None
    assert ub["volume_usd"] == 300.0                       # USD value still correct


def test_missing_amount_gives_none_price_not_zero():
    d = _swap(config.SOL_ADDRESS, "TOKEN_A")
    d["received"]["qty"] = None
    assert parse_wallet_tx(d, DEX)["price_per_token"] is None


def test_token_to_token_and_non_swap():
    assert parse_wallet_tx(_swap("TOKEN_B", "TOKEN_A"), DEX)["direction"] == "TOKEN_SWAP"
    other = dict(_swap(config.SOL_ADDRESS, "TOKEN_A"), kind="transfer")
    row = parse_wallet_tx(other, DEX)
    assert row["direction"] == "OTHER" and row["tx_type"] == "transfer"


def test_unknown_program_falls_back_to_source_name():
    d = dict(_swap(config.SOL_ADDRESS, "TOKEN_A"), program="UNKNOWN")
    assert parse_wallet_tx(d, DEX)["dex_program"] == "example_amm"


def _token_stats():
    return {"payload": {"n_holders": 100, "mcap": 50000.0, "fdv_usd": 60000.0, "liq": 3000.0,
                        "supply_total": 1e9, "supply_circ": 1e9, "px": 1e-5,
                        "buys_1m": 10, "sells_1m": 5, "vol_1m": 1234.5, "vol_buy_1m": 800.0,
                        "vol_sell_1m": 434.5, "vol_chg_1m": 12.3, "buys_24h": 1000, "vol_24h": 99999.0,
                        "title": "Example", "ticker": "EX", "dp": 6, "logo": "logo.png",
                        "extra": {"tw": "handle", "web": "site", "about": "an example token"}}}


def test_stats_merge_and_no_clobber():
    hp = {"payload": {"tok": {"top10": {"pct": 42.5}},
                      "groups": [{"group": "bundler", "n": 404, "pct": 50.45},
                                 {"group": "sniper", "n": 0, "pct": 0},
                                 {"group": "dev", "n": 1, "pct": 0.1}]}}
    md = {"payload": {"launch": {"done": False, "progress": 43.1,
                                 "curve": {"sol_real": "14000000000", "tok_real": "451000000000000",
                                           "tok_virtual": "731000000000000"}}}}
    s = sp.build_stats(_token_stats(), hp, md)
    assert s["holder"] == 100 and s["market_cap"] == 50000.0 and s["fdv"] == 60000.0
    assert s["buy_1m"] == 10 and s["sell_1m"] == 5 and s["buy_24h"] == 1000
    assert s["volume_1m_usd"] == 1234.5 and s["volume_buy_1m_usd"] == 800.0
    assert s["volume_1m_change_percent"] == 12.3
    assert s["top10_holder_percentage"] == 42.5
    assert s["bundler_count"] == 404 and s["bundler_held_percentage"] == 50.45
    assert s["sniper_count"] == 0 and s["dev_count"] == 1
    assert "insider_count" not in s                        # absent tag -> None -> dropped
    assert s["graduated"] == 0 and s["progress_percent"] == 43.1
    assert s["real_sol_reserves"] == "14000000000"         # reserves stay TEXT
    # a failed token-stats poll must not clobber cached values: build with only the holder breakdown
    assert "holder" not in sp.build_stats(holder=hp)


def test_identity_from_token_stats():
    ident = sp.identity_from_token_stats(_token_stats())
    assert ident["token_name"] == "Example" and ident["token_symbol"] == "EX" and ident["decimals"] == 6
    assert ident["twitter"] == "handle" and ident["description"] == "an example token"
    assert sp.identity_from_token_stats({"payload": {}}) == {} and sp.identity_from_token_stats(None) == {}
    assert sp.identity_from_token_stats({"payload": {"title": "X"}}) == {"token_name": "X"}


def test_batch_price_drops_missing_values():
    resp = {"payload": {"TOKEN_A": {"px": 0.5}, "TOKEN_B": {"px": None}, "TOKEN_C": "bad"}}
    assert sp.from_batch_price(resp) == {"TOKEN_A": 0.5}
    assert sp.from_batch_price(None) == {}


def test_fill_templates_keep_types():
    assert config.sub_wallet("W") == {"kind": "WATCH_WALLET", "body": {"wallet": "W"}}
    assert config.sub_price("T") == {"kind": "WATCH_PRICE", "body": {"token": "T", "bar": "fine"}}
    assert config.fill({"n": "{limit}", "s": "x-{limit}"}, limit=7) == {"n": 7, "s": "x-7"}


def test_fee_tiers_and_net_pnl():
    tiers = fees.DEX_FEE_TIERS
    assert fees.dex_fee_rate(tiers[0][0] - 1) == tiers[0][1]
    assert fees.dex_fee_rate(tiers[0][0]) == tiers[1][1]
    assert fees.dex_fee_rate(10 ** 12) == tiers[-1][1] and fees.dex_fee_rate(None) == tiers[-1][1]
    assert abs(fees.net_pnl_sol(1.0, 0.01, 0.02) - 0.97) < 1e-9
    assert abs(fees.net_pnl_usd(10.0, 0.25) - 9.75) < 1e-9 and fees.net_pnl_usd(5.0) == 5.0
    assert abs(fees.effective_cost_frac(1.02, 1.0) - 0.02) < 1e-9
    assert fees.effective_cost_frac(1.0, None) is None


def test_rest_client_batch_price_credits_and_chunking():
    import asyncio
    import math

    from collector import rest_client
    from collector.credits import CreditTracker

    calls = []

    class _Client(rest_client.RestClient):
        async def _get(self, path, params, credit_label, credit_est):
            calls.append((path, params, credit_est))
            self._credits.add(credit_label, credit_est)
            return {"payload": {t: {"px": 1.0} for t in params["tokens"].split(",")}}

    credits = CreditTracker()
    out = asyncio.run(_Client(credits).batch_price(["A", "B", "C", "D"]))      # chunk size 3 in test config
    assert [c[1]["tokens"] for c in calls] == ["A,B,C", "D"]
    assert [c[2] for c in calls] == [float(math.ceil(3 ** 0.5 * 1.0)), float(math.ceil(1 ** 0.5 * 1.0))]
    assert set(out["payload"]) == {"A", "B", "C", "D"} and credits.total == sum(c[2] for c in calls)
