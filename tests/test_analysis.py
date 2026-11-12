"""Analysis building blocks that can be checked without the private data:
cross-wallet lead/follow (C5), Kaplan-Meier with censoring (C3) and the causality of
the snapshot feature factory. All inputs are tiny synthetic frames.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from analysis import config, cross_wallet, exit as exit_mod, features  # noqa: E402


def _events():
    # LEAD enters every token first; FOL follows 100 s later; SOLO trades alone.
    rows = []
    for i, tok in enumerate(["T1", "T2", "T3"]):
        rows.append({"token_address": tok, "wallet": "LEAD", "timestamp": 1000 * i, "label": "ENTRY"})
        rows.append({"token_address": tok, "wallet": "FOL", "timestamp": 1000 * i + 100, "label": "ENTRY"})
    rows.append({"token_address": "T9", "wallet": "SOLO", "timestamp": 5, "label": "ENTRY"})
    rows.append({"token_address": "T1", "wallet": "LEAD", "timestamp": 50, "label": "EXIT"})   # ignored
    return pd.DataFrame(rows)


def test_overlap_and_lead_follow():
    fe = cross_wallet.first_entries(_events())
    ov = cross_wallet.overlap(fe)
    assert ov.loc["LEAD", "LEAD"] == 3 and ov.loc["LEAD", "FOL"] == 3 and ov.loc["LEAD", "SOLO"] == 0

    lf, dr = cross_wallet.lead_follow(fe)
    pair = lf[(lf["leader"] == "LEAD") & (lf["follower"] == "FOL")].iloc[0]
    assert pair["n_shared"] == 3 and pair["a_before_b"] == 1.0 and pair["median_lead_lag_s"] == 100
    rev = lf[(lf["leader"] == "FOL") & (lf["follower"] == "LEAD")].iloc[0]
    assert rev["a_before_b"] == 0.0
    assert not ((lf["leader"] == "SOLO") | (lf["follower"] == "SOLO")).any()      # below the pair minimum
    assert dr.index[0] == "LEAD" and dr.loc["LEAD", "first_mover_rate"] == 1.0
    assert dr.loc["FOL", "avg_entry_rank"] == 1.0


def test_head_to_head_lead_vs_follow():
    fe = cross_wallet.first_entries(_events())
    pos = pd.DataFrame([
        {"token_address": "T1", "wallet": "LEAD", "net_pnl_usd": 5.0},
        {"token_address": "T1", "wallet": "FOL", "net_pnl_usd": 1.0},
        {"token_address": "T2", "wallet": "LEAD", "net_pnl_usd": -2.0},
        {"token_address": "T2", "wallet": "FOL", "net_pnl_usd": 3.0},
        {"token_address": "T9", "wallet": "SOLO", "net_pnl_usd": 9.0},     # not shared: excluded
    ])
    h = cross_wallet.head_to_head(pos, fe)
    assert set(h.index) == {"LEAD", "FOL"}
    assert h.loc["LEAD", "shared_tokens"] == 2 and h.loc["LEAD", "win_rate"] == 0.5
    assert h.loc["LEAD", "median_pnl_when_LED"] == 1.5
    assert h.loc["FOL", "median_pnl_when_FOLLOWED"] == 2.0


def test_kaplan_meier_counts_censored_positions():
    # 4 holds: exits at 10 and 20, two still open at 30 (right-censored).
    km = exit_mod.km_estimate(np.array([10, 20, 30, 30]), np.array([1, 1, 0, 0]))
    s = dict(zip(km["t"], km["S"]))
    assert s[10] == pytest.approx(0.75) and s[20] == pytest.approx(0.5) and s[30] == pytest.approx(0.5)
    assert exit_mod.km_median(km) == 20
    # ignoring the censored holds would have given a median of 10: the survivorship bias
    naive = exit_mod.km_estimate(np.array([10, 20]), np.array([1, 1]))
    assert exit_mod.km_median(naive) == 10


def test_censored_positions_and_partial_exit_accounting():
    ev = pd.DataFrame([
        {"wallet": "A", "token_address": "T1", "timestamp": 0, "label": "ENTRY", "event_id": "1"},
        {"wallet": "A", "token_address": "T1", "timestamp": 5, "label": "PARTIAL_EXIT", "event_id": "2"},
        {"wallet": "A", "token_address": "T1", "timestamp": 9, "label": "EXIT", "event_id": "3"},
        {"wallet": "A", "token_address": "T2", "timestamp": 40, "label": "ENTRY", "event_id": "4"},
    ])
    cens = exit_mod.censored_positions(ev, end_t=100)
    assert len(cens) == 1 and cens.iloc[0]["token_address"] == "T2" and cens.iloc[0]["hold_sec"] == 60
    acc = exit_mod.partial_exit_accounting(ev)
    assert acc.loc["A", "partial_exit_rate"] == 0.5


def test_snapshot_features_are_causal():
    cols = config.all_snapshot_cols()
    ts = np.array([0, 100, 200, 300])
    frame = pd.DataFrame({c: np.zeros(len(ts)) for c in cols if c not in ("token_address", "timestamp")})
    frame["token_address"], frame["timestamp"] = "T1", ts
    frame["holder"] = [10, 20, 30, 40]
    frame["volume_buy_1m_usd"], frame["volume_sell_1m_usd"] = [3, 3, 3, 3], [1, 1, 1, 1]

    f = features.snapshot_features(frame, np.array([150, 300]))
    assert list(f["holder"]) == [20, 40]                   # state at-or-before the query time, never after
    short = config.DELTA_FIELDS["holder"][0]               # 60 s in the test config
    assert list(f[f"holder_d{short}"]) == [10, 10]         # 20 - 10 (as of t=90), 40 - 30 (as of t=240)
    assert f["sell_frac_1m"].iloc[0] == pytest.approx(0.25)
    assert set(config.delta_feature_names()) <= set(features.feature_names())


def test_exit_feature_names_carry_no_window_lengths():
    names = exit_mod.STATE_DYN + exit_mod.NONMONO_PATH + exit_mod.PATH_FEATS
    assert not any(any(ch.isdigit() for ch in n.replace("top10", "").replace("_1m", "").replace("_5m", ""))
                   for n in names), names


def test_first_touch_entry_uses_the_first_snapshot_after_it():
    """Documented exception to causality: when nothing was recorded before an entry
    (a wallet's first trade of a token), the earliest state is the first snapshot after it."""
    cols = config.all_snapshot_cols()
    ts = np.array([100, 200])
    frame = pd.DataFrame({c: np.zeros(len(ts)) for c in cols if c not in ("token_address", "timestamp")})
    frame["token_address"], frame["timestamp"] = "T1", ts
    frame["holder"] = [7, 9]
    f = features.snapshot_features(frame, np.array([90]))
    assert f["holder"].iloc[0] == 7


def test_blob_parsers_map_provider_keys_to_neutral_names():
    import json
    bf = json.dumps({"payload": {"rows": [{"op": 1, "hi": 2, "lo": 0.5, "cl": 1.5, "vol": 10, "ts": 60}]}})
    b = features.parse_backfill(bf)
    assert list(b[["o", "h", "l", "c", "v", "t"]].iloc[0]) == [1, 2, 0.5, 1.5, 10, 60]
    tt = json.dumps({"payload": {"rows": [{"who": "X", "pnl": 5.0, "vol": 100.0}]}})
    t = features.parse_top_traders(tt)
    assert t.loc[0, "owner"] == "X" and t.loc[0, "realized_pnl"] == 5.0
    rt = json.dumps([{"ts": 1, "way": "S", "who": "Y", "venue": "v", "leg_in": {"qty": 3},
                      "leg_out": {"qty": -2}, "quote_px": 10.0, "px": 0.1}])
    r = features.parse_recent_trades(rt)
    assert r.loc[0, "usd"] == 20.0 and r.loc[0, "side"] == "S" and r.loc[0, "base_amt"] == 3
    assert features.parse_backfill(None).empty and features.parse_recent_trades(None).empty


def test_attach_onchain_does_not_duplicate_holder_deltas():
    cols = config.all_snapshot_cols()
    ts = np.arange(0, 1000, 10)
    frame = pd.DataFrame({c: np.ones(len(ts)) for c in cols if c not in ("token_address", "timestamp")})
    frame["token_address"], frame["timestamp"] = "T1", ts
    panel = pd.DataFrame({"wallet": "A", "token_address": "T1", "position_id": 0, "entry_time": 500,
                          "t_in_trade": [0, 10, 20], "holder_d_short": 0.0, "label": [0, 0, 1]})
    out = exit_mod.attach_onchain(panel, {"T1": frame})
    assert not set(config.delta_feature_names(["holder"])) & set(out.columns)
    assert len(out) == 3 and "liquidity" in out.columns
