"""C4 — position sizing, on the full feature inventory. Fixed or proportional? To what?

Three reads per wallet:
  stats      : size distribution + consistency (low CV = narrow clip; NOT itself fixed-vs-prop).
  drivers    : does entry size scale with the on-chain state? spearman of log-size vs EVERY
               feature + a GroupKFold-by-token GBM R^2 (how much size variance the FULL state
               explains) + a shuffle-null (must collapse to ~0) + per-FAMILY R^2 attribution
               (mcap/structural vs the order-flow ladder vs classifications vs enrichment).
  conviction : do BIGGER positions earn more (size up on better setups) or not?

Why the three reads: low dispersion and fixed sizing are different claims. A narrow
size distribution can still scale with the token's state, so sizing is only called
"fixed" when it is both narrow and unexplained by state.

Size = the buy's volume_usd. Run `python -m analysis.sizing`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

from . import config, data, features

warnings.filterwarnings("ignore")

P = config.P


def size_stats(pos: pd.DataFrame) -> pd.DataFrame:
    p = pos[pos["tracked"] & pos["size_usd"].notna() & (pos["size_usd"] > 0)]
    rows = []
    for w, g in p.groupby("wallet"):
        s = g["size_usd"]
        rows.append({"wallet": w, "n": len(s), "med$": s.median(),
                     "cv": s.std() / s.mean(), "p25$": s.quantile(.25), "p75$": s.quantile(.75),
                     "p90/p10": s.quantile(.9) / max(s.quantile(.1), 1e-9)})
    return pd.DataFrame(rows).set_index("wallet")


def _cv_r2(g: pd.DataFrame, feats: list[str], target: str) -> float:
    """GBM R^2, GroupKFold by TOKEN (a token's entries never span train/test)."""
    folds = P("cv_folds")
    if g["token_address"].nunique() < folds or len(g) < P("sizing.min_rows") or not feats:
        return np.nan
    m = P("sizing.model")                   # depth / iterations: tuned privately
    reg = HistGradientBoostingRegressor(max_depth=m["max_depth"], max_iter=m["max_iter"], random_state=0)
    return float(cross_val_score(reg, g[feats], g[target], groups=g["token_address"],
                                 cv=GroupKFold(folds), scoring="r2").mean())


def size_drivers(ent: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in features.feature_names() if c in ent.columns]
    e = ent[ent["covered"] & ent["volume_usd"].notna() & (ent["volume_usd"] > 0)].copy()
    e["logsize"] = np.log(e["volume_usd"])
    min_entries, min_obs, min_unique = (P("sizing.min_wallet_entries"),
                                        P("sizing.correlate_min_obs"), P("sizing.correlate_min_unique"))
    groups = features.feature_groups()
    rows = []
    for w, g in e.groupby("wallet"):
        if len(g) < min_entries:
            continue
        corrs = []
        for f in feats:
            x = g[f]; m = x.notna()
            if m.sum() < min_obs or x[m].nunique() < min_unique:
                continue
            rho = spearmanr(x[m], g.loc[m, "logsize"]).statistic
            if rho == rho:
                corrs.append((f, rho))
        corrs.sort(key=lambda r: -abs(r[1]))
        r2 = _cv_r2(g, feats, "logsize")
        g_null = g.assign(_y=np.random.default_rng(0).permutation(g["logsize"].to_numpy()))
        null_r2 = _cv_r2(g_null, feats, "_y")
        fam = {name: _cv_r2(g, [f for f in cols if f in feats], "logsize")
               for name, cols in groups.items()}
        rows.append({"wallet": w, "size_R2": r2, "null_R2": null_r2,
                     **{f"R2_{k}": v for k, v in fam.items()},
                     "top_correlates": "; ".join(f"{f}{'+' if r > 0 else '-'}{abs(r):.2f}"
                                                 for f, r in corrs[:P("sizing.report_top_n")])})
    return pd.DataFrame(rows).set_index("wallet")


def conviction(pos: pd.DataFrame) -> pd.DataFrame:
    """Do bigger positions earn more? corr(size, return) + return by size quartile."""
    p = pos[pos["tracked"] & pos["size_usd"].notna() & (pos["size_usd"] > 0) & pos["net_pnl_usd"].notna()].copy()
    p["ret_pct"] = p["net_pnl_usd"] / p["size_usd"] * 100
    rows = []
    for w, g in p.groupby("wallet"):
        rho = spearmanr(g["size_usd"], g["ret_pct"]).statistic
        q = pd.qcut(g["size_usd"], 4, labels=["Q1sm", "Q2", "Q3", "Q4big"], duplicates="drop")
        by_q = g.groupby(q, observed=True)["ret_pct"].median()
        rows.append({"wallet": w, "corr_size_ret": rho,
                     "ret_small%": by_q.get("Q1sm", np.nan), "ret_big%": by_q.get("Q4big", np.nan)})
    return pd.DataFrame(rows).set_index("wallet")


def _self_test() -> None:
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    pos = data.reconstruct_positions(data.load_events())
    ent = features.entry_state_full(data.load_events(), features.load_snapshot_frames())
    groups = features.feature_groups()

    st = size_stats(pos)
    print("=== C4a  SIZE DISTRIBUTION (buy volume_usd) ===")
    print(st.round(2).to_string())
    st.round(3).to_csv(config.OUTPUT_DIR / "c4_size_stats.csv")

    dr = size_drivers(ent)
    print("\n=== C4b  SIZE DRIVERS (full inventory; GroupKFold-by-token R^2, null must be ~0) ===")
    fam_cols = [f"R2_{k}" for k in groups]
    print(dr[["size_R2", "null_R2"] + fam_cols].round(3).to_string())
    print("\n  top size-correlates (spearman, full inventory):")
    for w, r in dr.iterrows():
        print(f"    {w:10} {r['top_correlates']}")
    dr.round(4).to_csv(config.OUTPUT_DIR / "c4_size_drivers.csv")

    cv = conviction(pos)
    print("\n=== C4c  CONVICTION (do bigger positions earn more?) ===")
    print(cv.round(3).to_string())
    cv.round(4).to_csv(config.OUTPUT_DIR / "c4_conviction.csv")

    # Verdict cut-offs (tuned privately): how far R^2 must clear its shuffle-null to call
    # sizing state-proportional, the dispersion below which it is a fixed clip, and the
    # |corr(size, return)| needed to call a conviction direction.
    v = P("sizing.verdict")
    print("\n=== verdict ===")
    for w in st.index:
        if w not in dr.index:
            continue
        r2 = dr.loc[w, "size_R2"]; null = dr.loc[w, "null_R2"]
        fam = {k: dr.loc[w, f"R2_{k}"] for k in groups}
        best = max(fam, key=lambda k: (fam[k] if fam[k] == fam[k] else -9))
        sizing = (f"proportional to STATE (R2={r2:.2f} vs null {null:.2f}; top family={best} {fam[best]:.2f})"
                  if (r2 == r2 and r2 - (null if null == null else 0) > v["min_r2_over_null"]) else
                  "FIXED clip" if st.loc[w, "cv"] < v["fixed_clip_cv_max"] else "weak/idiosyncratic")
        conv = cv.loc[w, "corr_size_ret"] if w in cv.index else np.nan
        skill = ("sizes-up-on-winners" if conv > v["conviction_min_abs_corr"]
                 else "sizes-down-on-winners" if conv < -v["conviction_min_abs_corr"]
                 else "size~outcome-neutral")
        print(f"  {w:10} {sizing:62} | {skill}")
    print("\nC4 OK - sizing decoded on the FULL inventory; written to analysis/outputs/.")


if __name__ == "__main__":
    _self_test()
