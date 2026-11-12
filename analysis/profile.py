"""C1 — per-wallet descriptive signature, on the full feature inventory.

Two purely-descriptive signatures (no modelling: that is C2/C4):
  behaviour    : hold-duration regime, position-size consistency, win-rate, profit factor
                 (round-trip OUTCOMES; these don't need on-chain features).
  entry-state  : the typical on-chain state at a buy, across the WHOLE feature
                 inventory (`features.entry_state_full`): a readable median signature PLUS a
                 per-wallet "most distinctive features" screen over ALL features (robust
                 median/MAD vs the cohort) so nothing is left unexamined.

Run `python -m analysis.profile`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, features

pd.set_option("display.width", 220, "display.max_columns", 50)

P = config.P

# a readable cross-family subset for the headline signature (the full screen is below)
READABLE = ["market_cap", "liquidity", "holder", "top10_holder_percentage", "progress_percent",
            "graduated", "sniper_held_percentage", "bundler_held_percentage",
            "insider_held_percentage", "dev_held_percentage", "n_trades_5m", "sell_frac_5m",
            "log_vol_5m", "net_flow_1m", "bf_pre_return", "tt_frac_profitable", "rt_sell_frac"]


def hold_regime(hold_med: float) -> str:
    """Bucket a median hold into ultra-fast / fast / medium / slow. The three
    boundaries (seconds) are tuned privately."""
    ultra, fast, medium = P("profile.hold_regime_bounds_sec")
    return ("ultra-fast" if hold_med < ultra else "fast" if hold_med < fast
            else "medium" if hold_med < medium else "slow")


def behaviour_signature(pos: pd.DataFrame) -> pd.DataFrame:
    """Per-wallet round-trip behaviour: hold regime, size consistency, PnL quality."""
    ultra, _fast, medium = P("profile.hold_regime_bounds_sec")
    rows = []
    for w, g in pos.groupby("wallet"):
        t = g[g["tracked"]]
        hold, size, pnl = t["hold_duration_sec"].dropna(), t["size_usd"].dropna(), g["net_pnl_usd"].dropna()
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        rows.append({
            "wallet": w, "positions": len(g), "tokens": g["token_address"].nunique(),
            "hold_p25": hold.quantile(.25), "hold_med": hold.median(), "hold_p75": hold.quantile(.75),
            "size_med$": size.median(), "size_cv": (size.std() / size.mean()) if size.mean() else np.nan,
            "win_rate": (pnl > 0).mean(), "net$": pnl.sum(), "med_pnl$": pnl.median(),
            "profit_factor": (wins.sum() / -losses.sum()) if losses.sum() < 0 else np.nan,
            "ultra_fast%": (hold < ultra).mean(), "slow%": (hold > medium).mean(),
        })
    return pd.DataFrame(rows).set_index("wallet").sort_values("net$", ascending=False)


def entry_signature_full(ent: pd.DataFrame) -> pd.DataFrame:
    """Per-wallet MEDIAN of the readable cross-family entry-state subset (+ coverage)."""
    feats = [f for f in READABLE if f in ent.columns]
    return ent.groupby("wallet")[feats].median()


def distinctive_features(ent: pd.DataFrame, k: int | None = None) -> dict[str, list]:
    """For each wallet, the features whose ENTRY-median deviates most from the cohort (all
    wallets' entries), scored by robust z = (wallet_med - cohort_med) / cohort_MAD. Screens the
    ENTIRE inventory so the signature reflects all data, not a hand-picked subset."""
    if k is None:
        k = P("profile.distinctive_top_k")
    feats = [c for c in ent.columns if c not in
             ("wallet", "token_address", "timestamp", "label", "volume_usd", "covered")
             and pd.api.types.is_numeric_dtype(ent[c])]
    cohort_med = ent[feats].median()
    cohort_mad = (ent[feats] - cohort_med).abs().median()
    # skip NEAR-CONSTANT features (tiny MAD relative to level inflates z spuriously, e.g.
    # total supply) but keep genuinely sparse ones (median ~0: reserves, sell-side).
    min_rel_mad = P("profile.min_relative_mad")
    usable = [f for f in feats if cohort_mad[f] and cohort_mad[f] > 0
              and (abs(cohort_med[f]) < 1e-9 or cohort_mad[f] / abs(cohort_med[f]) > min_rel_mad)]
    out = {}
    for w, g in ent.groupby("wallet"):
        wm = g[usable].median()
        z = ((wm - cohort_med[usable]) / cohort_mad[usable]).abs().sort_values(ascending=False)
        out[w] = [(f, wm[f], cohort_med[f]) for f in z.head(k).index]
    return out


def _self_test() -> None:
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    ev = data.load_events()
    pos = data.reconstruct_positions(ev)

    beh = behaviour_signature(pos)
    print("=== C1a  BEHAVIOUR SIGNATURE (round-trips) ===")
    print(beh.round(2).to_string())
    beh.round(4).to_csv(config.OUTPUT_DIR / "c1_behaviour.csv")

    print("\nbuilding full entry-state matrix ...")
    ent = features.entry_state_full(ev, features.load_snapshot_frames())
    ent = ent[ent["covered"]]
    med = entry_signature_full(ent)
    print(f"\n=== C1b  ENTRY-STATE SIGNATURE (median, readable subset; coverage {ent['covered'].mean():.0%}) ===")
    print(med.round(2).to_string())
    med.round(4).to_csv(config.OUTPUT_DIR / "c1_entry_state.csv")

    print("\n=== C1c  MOST DISTINCTIVE ENTRY FEATURES (full-inventory screen vs cohort) ===")
    dist = distinctive_features(ent)
    rows = []
    for w in sorted(dist):
        print(f"\n  --- {w} ---")
        for f, wm, cm in dist[w]:
            d = "HIGHER" if wm > cm else "lower"
            print(f"     {f:26} wallet {wm:>12.4g} vs cohort {cm:>12.4g}  -> {d}")
            rows.append({"wallet": w, "feature": f, "wallet_med": wm, "cohort_med": cm})
    pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / "c1_distinctive.csv", index=False)

    print("\n=== quick read ===")
    narrow_cv = P("profile.narrow_size_cv_max")
    for w in beh.index:
        b = beh.loc[w]
        spread = "narrow-$" if b["size_cv"] < narrow_cv else "wide-$"
        print(f"  {w:10} {hold_regime(b['hold_med']):10} hold~{b['hold_med']:.0f}s | {spread} ~${b['size_med$']:.0f} "
              f"(cv {b['size_cv']:.2f}) | win {b['win_rate']*100:.0f}% PF {b['profit_factor']:.2f} | net ${b['net$']:+.0f}")
    print("\nC1 OK - behaviour + full-inventory entry signature written to analysis/outputs/.")


if __name__ == "__main__":
    _self_test()
