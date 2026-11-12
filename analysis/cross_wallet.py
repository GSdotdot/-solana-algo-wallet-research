"""C5 — cross-wallet: co-trading, lead/follow, relative performance, competitive field.

This is the step that tests the observation the project started from: several
automated wallets converging on the same tokens within a short window.

Behavioural layer (the tracked wallets' own trades):
  overlap      : wallet x wallet shared-token counts (Jaccard-style overlap matrix).
  lead_follow  : pairwise P(A enters before B) + median lag on shared tokens; per-wallet
                 discovery rank (avg normalised entry-order position) + first-mover rate.
  head_to_head : on CO-TRADED tokens, each wallet's realised net PnL, win-rate vs the others,
                 and whether LEADING (entered first) out-earns FOLLOWING.
Competitive layer (the top-traders enrichment blob: the rest of the field):
  competitive  : on tokens where a tracked wallet is a top trader, its rank + realised PnL vs
                 the OTHER top traders (beats-median rate, top-1 rate), and the recurring
                 untracked owners that show up alongside the tracked wallets (candidate
                 related algorithms).

Rigor: medians + n reported everywhere; low-n pairs dropped; shared-token + top-traders
(point-in-time snapshot) selection caveats stated. Run `python -m analysis.cross_wallet`.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from . import config, data, features

OPEN = ("ENTRY", "FLIP")
MIN_PAIR = config.P("cross_wallet.min_shared_tokens")   # min shared tokens to report a wallet pair; tuned privately


def first_entries(events: pd.DataFrame) -> pd.DataFrame:
    """Per (token, wallet) first-entry timestamp."""
    ent = events[events["label"].isin(OPEN)]
    return ent.groupby(["token_address", "wallet"])["timestamp"].min().reset_index()


def overlap(fe: pd.DataFrame) -> pd.DataFrame:
    toks = {w: set(g["token_address"]) for w, g in fe.groupby("wallet")}
    ws = sorted(toks)
    mat = pd.DataFrame(index=ws, columns=ws, dtype=float)
    for a in ws:
        for b in ws:
            mat.loc[a, b] = len(toks[a]) if a == b else len(toks[a] & toks[b])
    return mat


def lead_follow(fe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    piv = fe.pivot_table(index="token_address", columns="wallet", values="timestamp", aggfunc="min")
    rows = []
    for a, b in permutations(sorted(fe["wallet"].unique()), 2):
        both = piv[[a, b]].dropna()
        if len(both) < MIN_PAIR:
            continue
        lag = (both[b] - both[a])                              # >0 => a entered before b
        rows.append({"leader": a, "follower": b, "n_shared": len(both),
                     "a_before_b": float((lag > 0).mean()),
                     "median_lead_lag_s": float(lag[lag > 0].median()) if (lag > 0).any() else np.nan})
    lf = pd.DataFrame(rows)
    # per-wallet discovery rank (0 = first into a token, 1 = last) + first-mover rate
    drank = []
    for _, row in piv.iterrows():
        present = row.dropna().sort_values()
        if len(present) < 2:
            continue
        for r, w in zip(np.linspace(0, 1, len(present)), present.index):
            drank.append({"wallet": w, "norm_rank": r, "first": r == 0})
    dr = (pd.DataFrame(drank).groupby("wallet")
          .agg(cotraded=("norm_rank", "size"), avg_entry_rank=("norm_rank", "mean"),
               first_mover_rate=("first", "mean")).sort_values("avg_entry_rank"))
    return lf, dr


def head_to_head(pos: pd.DataFrame, fe: pd.DataFrame) -> pd.DataFrame:
    """On co-traded tokens: realised-PnL win-rate vs the others + lead-vs-follow profitability."""
    nw = pos.groupby("token_address")["wallet"].nunique()
    shared = set(nw[nw >= 2].index)
    wt = (pos[pos["token_address"].isin(shared)]
          .groupby(["token_address", "wallet"])["net_pnl_usd"].sum().reset_index())
    order = fe.set_index(["token_address", "wallet"])["timestamp"]
    rows = []
    for tok, g in wt.groupby("token_address"):
        g = g.sort_values("net_pnl_usd", ascending=False).reset_index(drop=True)
        firsts = order.loc[tok] if tok in order.index.get_level_values(0) else None
        lead_w = firsts.idxmin() if firsts is not None and len(firsts) else None
        for rank, r in g.iterrows():
            rows.append({"wallet": r["wallet"], "pnl": r["net_pnl_usd"],
                         "norm_rank": rank / (len(g) - 1) if len(g) > 1 else 0.0,
                         "win": rank == 0, "led": r["wallet"] == lead_w})
    d = pd.DataFrame(rows)
    agg = d.groupby("wallet").agg(shared_tokens=("pnl", "size"), win_rate=("win", "mean"),
                                  avg_norm_rank=("norm_rank", "mean"), median_pnl=("pnl", "median"))
    led = d[d["led"]].groupby("wallet")["pnl"].median().rename("median_pnl_when_LED")
    fol = d[~d["led"]].groupby("wallet")["pnl"].median().rename("median_pnl_when_FOLLOWED")
    return agg.join(led).join(fol)


def competitive(events: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Top-traders blob: the tracked wallets' standing vs the rest of the field, plus
    the untracked owners that recur alongside them. Untracked owners are returned as
    anonymous ranks (EXT_WALLET_01, ...): their addresses never leave this function."""
    con = data._conn()
    ti = pd.read_sql("SELECT token_address, top_traders_json FROM token_identity", con)
    con.close()
    targets = config.WALLET_LABELS
    rows, allies = [], {}
    for r in ti.itertuples():
        tt = features.parse_top_traders(r.top_traders_json)
        if not len(tt) or "owner" not in tt.columns:
            continue
        tt = tt.assign(pnl=pd.to_numeric(tt.get("realized_pnl", np.nan), errors="coerce")) \
               .sort_values("pnl", ascending=False).reset_index(drop=True)
        mine = tt[tt["owner"].isin(targets)]
        if not len(mine):
            continue
        others_pnl = tt[~tt["owner"].isin(targets)]["pnl"]
        for _, o in mine.iterrows():
            rk = int(tt.index[tt["owner"] == o["owner"]][0])
            rows.append({"wallet": targets[o["owner"]], "rank": rk, "n_top": len(tt),
                         "our_pnl": o["pnl"], "is_top1": rk == 0,
                         "beats_median_other": (o["pnl"] > others_pnl.median()) if len(others_pnl) else np.nan})
        for own in tt[~tt["owner"].isin(targets)]["owner"]:
            allies[own] = allies.get(own, 0) + 1
    comp = pd.DataFrame(rows)
    agg = comp.groupby("wallet").agg(
        tokens_as_topN=("rank", "size"), median_rank=("rank", "median"),
        top1_rate=("is_top1", "mean"), beats_other_rate=("beats_median_other", "mean"),
        median_our_pnl=("our_pnl", "median")).sort_values("beats_other_rate", ascending=False)
    ranked = sorted(allies.items(), key=lambda x: -x[1])[:config.P("cross_wallet.top_recurring_n")]
    top_allies = [(f"EXT_WALLET_{i:02d}", n) for i, (_own, n) in enumerate(ranked, 1)]
    return agg, top_allies


def _self_test() -> None:
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    ev = data.load_events()
    pos = data.reconstruct_positions(ev)
    fe = first_entries(ev)

    print("=== C5a  CO-TRADING OVERLAP (shared tokens; diag = wallet's own token count) ===")
    ov = overlap(fe)
    print(ov.astype(int).to_string())
    ov.to_csv(config.OUTPUT_DIR / "c5_overlap.csv")

    lf, dr = lead_follow(fe)
    print("\n=== C5b  DISCOVERY RANK (who enters co-traded tokens FIRST) ===")
    print(dr.round(3).to_string())
    strong_n = config.P("cross_wallet.strong_pair.min_shared_tokens")     # tuned privately
    strong_rate = config.P("cross_wallet.strong_pair.min_lead_rate")      # tuned privately
    print("\n  strongest lead/follow pairs:")
    strong = lf[(lf["n_shared"] >= strong_n) & (lf["a_before_b"] >= strong_rate)] \
        .sort_values("a_before_b", ascending=False)
    for _, r in strong.iterrows():
        print(f"    {r['leader']:10} leads {r['follower']:10}  {r['a_before_b']*100:>3.0f}% of {int(r['n_shared'])} shared "
              f"| median lag {r['median_lead_lag_s']/60:>5.1f} min")
    lf.round(4).to_csv(config.OUTPUT_DIR / "c5_lead_follow.csv")
    dr.round(4).to_csv(config.OUTPUT_DIR / "c5_discovery.csv")

    h2h = head_to_head(pos, fe)
    print("\n=== C5c  HEAD-TO-HEAD on shared tokens (realised net PnL) + lead-vs-follow ===")
    print(h2h.round(3).to_string())
    h2h.round(4).to_csv(config.OUTPUT_DIR / "c5_head_to_head.csv")

    comp, allies = competitive(ev)
    print("\n=== C5d  COMPETITIVE FIELD (top traders: tracked wallets vs the rest) ===")
    print(comp.round(3).to_string())
    print("\n  recurring UNTRACKED owners co-present with the tracked wallets (candidate related algorithms):")
    for own, n in allies:
        print(f"    {own}  x{n} tokens")
    comp.round(4).to_csv(config.OUTPUT_DIR / "c5_competitive.csv")

    print("\n=== read ===")
    print(f"  discoverer (lowest avg entry-rank): {dr.index[0]} ({dr['avg_entry_rank'].iloc[0]:.2f}); "
          f"latest follower: {dr.index[-1]} ({dr['avg_entry_rank'].iloc[-1]:.2f})")
    print("  CAVEATS: shared-token stats are a selected subset; the top-traders blob is a"
          " point-in-time snapshot (realised PnL not necessarily final).")
    print("\nC5 OK - cross-wallet behavioural + competitive written to analysis/outputs/.")


if __name__ == "__main__":
    _self_test()
