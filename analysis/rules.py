"""C7 — synthesis: per-wallet decoded behaviour + cross-wallet picture + signal-vs-noise.

Reads the persisted C1–C6 outputs (run those first) and ties them into the deliverable:
  * per-wallet STRATEGY CARD (behaviour, entry regime + TA, exit logic, sizing, ecosystem role),
  * the CROSS-WALLET picture (the discover -> pile-in ordering; universal vs wallet-specific),
  * a SIGNAL-vs-NOISE table: which signal categories actually carry out-of-sample predictive
    power across modules (C2 entry group-fidelity, C3 within-hold family lift, C4 sizing
    family R^2, C6 indicator AUC) vs which are noise,
  * confidence + caveats (single market regime; selected subsets; point-in-time enrichment).

PUBLIC-COPY NOTE. The private version of this module also carries hand-written
conclusions: a verdict per signal category, a paragraph per wallet, and the
"universal vs wallet-specific" summary. Those are the findings of the research, so
they are removed here and marked `[redacted: finding]`. Everything that is computed
(the tables, the cards, the ordering) is kept as code.

Writes `analysis/outputs/c7_report.md`. Run `python -m analysis.rules`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .profile import hold_regime

OUT = config.OUTPUT_DIR
P = config.P
REDACTED_FINDING = "[redacted: finding]"

NEED = ["c1_behaviour", "c1_distinctive", "c2_fidelity", "c2_discriminators", "c3_verdict",
        "c3_within_hold", "c4_size_drivers", "c4_conviction", "c5_discovery", "c5_head_to_head",
        "c5_competitive", "c5_lead_follow", "c6_entry_sweep_full", "c6_exit_sweep_full"]

# Indicator families grouped into the signal categories compared in the table.
INDICATOR_CATEGORIES = {
    "order-flow / volume": ["vol_roc", "vol_zscore", "obv_slope", "force_index", "eom"],
    "volatility (bb/std/atr)": ["std_norm", "bb_width", "atr_norm", "ulcer", "bb_pctb", "keltner_pos"],
    "price-path / momentum": ["frlow", "frhigh", "roc", "momentum", "wma_dist", "hma_dist",
                              "rsi", "stoch_k", "cci"],
    "smoothed oscillators (tsi/macd/cmf/rvi)": ["tsi", "macd_hist", "cmf", "rvi"],
}


def load() -> dict:
    out, missing = {}, []
    for name in NEED:
        p = OUT / f"{name}.csv"
        if p.exists():
            idx = 0 if name in ("c1_behaviour", "c3_verdict") else None
            out[name] = pd.read_csv(p, index_col=idx)
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"missing outputs (run the producing module first): {missing}")
    return out


def _md_table(df: pd.DataFrame) -> str:
    """GitHub-flavoured markdown table (no tabulate dependency)."""
    def cell(v):
        return "" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, float) else str(v))
    head = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def _fam(series: pd.Series) -> pd.Series:
    return series.str.replace(r"_\d+$", "", regex=True)


def c6_best(c6: pd.DataFrame, wallet: str, fams: list[str]) -> float:
    f = c6[(c6["wallet"] == wallet) & _fam(c6["indicator"]).isin(fams)]
    if "validated" in f:
        f = f[f["validated"]]
    return float((0.5 + (f["auc_test"] - 0.5).abs()).max()) if len(f) else np.nan


def c6_top(c6: pd.DataFrame, wallet: str, k: int | None = None) -> str:
    if k is None:
        k = P("rules.top_indicators_n")
    f = c6[c6["wallet"] == wallet].copy()
    if "validated" in f:
        f = f[f["validated"]]
    f = f.reindex(f["auc_train"].sub(0.5).abs().sort_values(ascending=False).index)
    return ", ".join(f["indicator"].head(k))


def ecosystem_role(avg_entry_rank: float | None) -> str:
    """Discoverer / mid-pack / late follower from the average normalised entry rank
    on co-traded tokens (0 = always first in, 1 = always last). The two cut-offs are
    tuned privately."""
    if avg_entry_rank is None:
        return "mid-pack"
    lo, hi = P("rules.role_rank_bounds")
    return "discoverer" if avg_entry_rank < lo else "late follower" if avg_entry_rank > hi else "mid-pack"


def _row(df: pd.DataFrame, w: str):
    return df.loc[df["wallet"] == w].squeeze() if w in df["wallet"].values else None


def wallet_card(w: str, o: dict) -> list[str]:
    b = o["c1_behaviour"].loc[w]
    disc, comp, h2h = _row(o["c5_discovery"], w), _row(o["c5_competitive"], w), _row(o["c5_head_to_head"], w)
    c2 = _row(o["c2_fidelity"], w)
    c3 = o["c3_verdict"].loc[w]
    c4 = _row(o["c4_size_drivers"], w)
    conv = o["c4_conviction"].loc[o["c4_conviction"]["wallet"] == w, "corr_size_ret"].squeeze()
    role = ecosystem_role(disc["avg_entry_rank"] if disc is not None else None)
    L = [f"### {w} — {role} ({hold_regime(b['hold_med'])}, hold ~{b['hold_med']:.0f}s)",
         f"- **Behaviour:** {int(b['positions'])} positions / {int(b['tokens'])} tokens · size ~${b['size_med$']:.0f} "
         f"(cv {b['size_cv']:.2f}) · win {b['win_rate']*100:.0f}% · PF {b['profit_factor']:.2f} · net ${b['net$']:+,.0f}",
         f"- **Entry (C2 fidelity {c2['fidelity']:.2f}, null {c2['null']:.2f}):** "
         f"grp orderflow {c2['grp_orderflow']:.2f} vs structural {c2['grp_structural']:.2f}; "
         f"top TA (C6): {c6_top(o['c6_entry_sweep_full'], w)}",
         f"- **Exit (C3):** {c3['verdict']} — within-hold decodability {c3['decodability']:.2f} "
         f"(price {c3['wp_price']:.2f} → +on-chain {c3['wp_onchain']:.2f})",
         f"- **Sizing (C4):** R²={c4['size_R2']:.2f} (null {c4['null_R2']:+.2f}); {c4['top_correlates']}; "
         f"conviction corr(size,ret)={conv:+.2f}"]
    if disc is not None:
        L.append(f"- **Ecosystem (C5):** entry-rank {disc['avg_entry_rank']:.2f} (first {disc['first_mover_rate']*100:.0f}%)"
                 + (f" · vs field: beats {comp['beats_other_rate']*100:.0f}% of other top-traders, median rank {comp['median_rank']:.0f}" if comp is not None else "")
                 + (f" · shared-token win {h2h['win_rate']*100:.0f}%" if h2h is not None else ""))
    return L


def signal_vs_noise(o: dict) -> pd.DataFrame:
    """One row per signal category; one column per module's out-of-sample measure.
    The `verdict` column is hand-written in the private version and redacted here."""
    c2, c3, c4 = o["c2_fidelity"], o["c3_within_hold"], o["c4_size_drivers"]
    c6e = o["c6_entry_sweep_full"]

    def c6fam(category):
        fams = INDICATOR_CATEGORIES[category]
        return np.nanmax([c6_best(c6e, w, fams) for w in c6e["wallet"].unique()])

    def lift(col):  # within-hold family lift over price, max across wallets
        return float((c3[col] - c3["wp_price"]).max())

    nan = np.nan
    rows = [
        ("order-flow / volume", c2["grp_orderflow"].max(), lift("wp_volume"), nan, c6fam("order-flow / volume")),
        ("volatility (bb/std/atr)", nan, nan, nan, c6fam("volatility (bb/std/atr)")),
        ("price-path / momentum", nan, (c3["wp_price"] - c3["wp_null"]).max(), nan, c6fam("price-path / momentum")),
        ("structural (mcap/liq/fdv)", c2["grp_structural"].max(), lift("wp_structural"), c4["R2_structural"].max(), nan),
        ("classifications (snipe/bundle/ins/dev)", c2["grp_classification"].max(), lift("wp_classif"), nan, nan),
        ("holder / liquidity dynamics", nan, lift("wp_holder_liq"), nan, nan),
        ("enrichment (top_traders/backfill)", c2["grp_enrichment"].max(), nan, nan, nan),
        ("smoothed oscillators (tsi/macd/cmf/rvi)", nan, nan, nan, c6fam("smoothed oscillators (tsi/macd/cmf/rvi)")),
    ]
    df = pd.DataFrame(rows, columns=["signal_category", "entry_AUC", "exit_lift", "sizing_R2", "indicator_AUC"])
    df["verdict"] = REDACTED_FINDING
    return df


def report(o: dict) -> str:
    disc = o["c5_discovery"].sort_values("avg_entry_rank")
    lf = o["c5_lead_follow"]
    leader = disc["wallet"].iloc[0]
    strong = lf[(lf["leader"] == leader) & (lf["a_before_b"] >= P("rules.consistent_lead_rate"))]
    lag_min = strong["median_lead_lag_s"] / 60
    md = ["# Wallet research — decoded behaviour (Module C synthesis)\n",
          "_Generated by `analysis/rules.py` from C1–C6 outputs. A single market regime; "
          "all model numbers are out-of-sample (GroupKFold-by-token / token-holdout) over the full "
          "data inventory._\n",
          "## 1. The ecosystem — a discover→pile-in ordering\n",
          f"Entry order on co-traded tokens runs **{' → '.join(disc['wallet'])}** "
          f"(earliest → latest). **{leader}** is first into a shared token "
          f"{disc['first_mover_rate'].iloc[0]*100:.0f}% of the time"
          + (f" and consistently leads {len(strong)} other wallet(s), by a median "
             f"{lag_min.min():.0f}–{lag_min.max():.0f} min.\n" if len(strong) else ".\n"),
          f"Interpretation: {REDACTED_FINDING}\n",
          "## 2. Per-wallet strategy cards\n"]
    order = o["c1_behaviour"].sort_values("net$", ascending=False).index
    for w in order:
        md += wallet_card(w, o) + [f"- **Read:** {REDACTED_FINDING}", ""]
    md += ["## 3. Universal vs wallet-specific\n",
           f"**Universal:** {REDACTED_FINDING}\n",
           f"**Wallet-specific:** {REDACTED_FINDING}\n",
           "## 4. Signal vs noise (what actually predicts, across modules)\n",
           _md_table(signal_vs_noise(o).round(2)), "\n",
           "## 5. Confidence & caveats\n",
           "- **One market regime**: cross-regime robustness is untested.\n"
           "- Cross-wallet head-to-head + lead/follow are a **selected subset** (co-traded tokens skew to notable "
           "tokens); the top-traders blob is a **point-in-time snapshot** (realised PnL not necessarily final).\n"
           "- The ENTRY TRIGGER for first-movers is below the data's resolution (discovery is wallet-driven, so "
           "logging starts AT the first entry); entry CONDITIONS are decoded, not the trigger itself.\n"
           "- C6 absolute AUCs are vs an all-bars baseline (mostly quiet) → partly 'they trade in active bars'; the "
           "RELATIVE family ranking is the signal.\n"]
    return "\n".join(md)


def _self_test() -> None:
    o = load()
    text = report(o)
    (OUT / "c7_report.md").write_text(text, encoding="utf-8")
    # console: signal-vs-noise + the per-wallet one-liners
    print("=== C7  SIGNAL vs NOISE (OOS predictive power across modules) ===")
    print(signal_vs_noise(o).round(2).to_string(index=False))
    print("\n=== C7  per-wallet (decoded), richest first ===")
    for w in o["c1_behaviour"].sort_values("net$", ascending=False).index:
        c3 = o["c3_verdict"].loc[w]
        d = o["c5_discovery"]
        rank = d.loc[d["wallet"] == w, "avg_entry_rank"]
        print(f"  {w:10} {ecosystem_role(rank.iloc[0] if len(rank) else None):13} exit: {c3['verdict']}")
    print("\nC7 OK - full report written to analysis/outputs/c7_report.md")


if __name__ == "__main__":
    _self_test()
