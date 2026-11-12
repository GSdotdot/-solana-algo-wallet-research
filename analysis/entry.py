"""C2 — entry conditions, decoded on the full feature inventory.

A first version of this step used a small hand-picked feature subset and a random
k-fold split, which leaks: rows from one token land in both train and test. This
rebuild:
  * consumes the FULL feature factory (`features.entry_state_full`: the whole
    on-chain ladder + trader-type classifications + the REST enrichments),
  * contrasts each wallet's entry-states vs a UNIVERSE baseline (random states of the same
    tracked tokens: the states the cohort's tokens passed through),
  * scores with **GroupKFold-BY-TOKEN** (a token's entries + baseline rows never span
    train/test) + a **label-shuffle null** (must collapse to ~0.50),
  * attributes decodability to feature GROUPS (structural / classification / order-flow ladder /
    enrichment) so we learn WHICH data actually distinguishes a buy, not an assumed subset.

Run `python -m analysis.entry`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config, data, features

warnings.filterwarnings("ignore")

P = config.P


def model_features() -> list[str]:
    """The full feature inventory, the starting universe for C2."""
    return features.feature_names()


def _hgb():
    m = P("entry.model")                    # depth / iterations / learning rate: tuned privately
    return HistGradientBoostingClassifier(max_depth=m["max_depth"], max_iter=m["max_iter"],
                                          learning_rate=m["learning_rate"],
                                          class_weight="balanced", random_state=0)


def universe_baseline(frames: dict, enr: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """Random (token, time) states of the tracked-token universe, with the SAME full features."""
    if n is None:
        n = P("entry.baseline_samples")     # tuned privately
    rng = np.random.default_rng(0)
    toks = list(frames)
    sizes = np.array([len(frames[t]) for t in toks], float)
    pick = rng.choice(len(toks), size=n, p=sizes / sizes.sum())
    parts = []
    for ti in np.unique(pick):
        fr = frames[toks[ti]]
        ts = fr["timestamp"].to_numpy()[rng.integers(0, len(fr), size=int((pick == ti).sum()))]
        f = features.snapshot_features(fr, ts)
        f["token_address"] = toks[ti]
        parts.append(f)
    return pd.concat(parts, ignore_index=True).merge(enr, on="token_address", how="left")


def cv_auc(X: pd.DataFrame, y: np.ndarray, g: pd.Series, feats: list[str]) -> float:
    folds, min_class = P("cv_folds"), P("entry.min_class_count")
    if g.nunique() < folds or y.sum() < min_class or (1 - y).sum() < min_class:
        return np.nan
    return float(cross_val_score(_hgb(), X[feats], y, groups=g, cv=GroupKFold(folds), scoring="roc_auc").mean())


def uni_auc(ent: pd.DataFrame, base: pd.DataFrame, feats: list[str]) -> list[tuple]:
    out = []
    min_obs = P("entry.univariate_min_obs")
    for f in feats:
        x = pd.concat([ent[f], base[f]], ignore_index=True)
        y = np.r_[np.ones(len(ent)), np.zeros(len(base))]
        m = x.notna().to_numpy()
        if m.sum() < min_obs or y[m].sum() in (0, m.sum()):
            continue
        try:
            a = roc_auc_score(y[m], x[m].to_numpy())
        except ValueError:
            continue
        out.append((f, a, ent[f].median(), base[f].median()))
    out.sort(key=lambda r: -abs(r[1] - 0.5))
    return out


def analyse(wallet: str, ent: pd.DataFrame, base: pd.DataFrame, feats: list[str]) -> dict:
    e = ent[ent["wallet"] == wallet]
    X = pd.concat([e[feats], base[feats]], ignore_index=True)
    y = np.r_[np.ones(len(e)), np.zeros(len(base))].astype(int)
    g = pd.concat([e["token_address"], base["token_address"]], ignore_index=True)
    fid = cv_auc(X, y, g, feats)
    null = cv_auc(X, pd.Series(np.random.default_rng(0).permutation(y)).to_numpy(), g, feats)
    grp = {name: cv_auc(X, y, g, [f for f in cols if f in feats])
           for name, cols in features.feature_groups().items()}
    t = P("entry.rule_tree")                # readable surrogate tree: depth / leaf size tuned privately
    tree = make_pipeline(SimpleImputer(strategy="median"),
                         DecisionTreeClassifier(max_depth=t["max_depth"], min_samples_leaf=t["min_samples_leaf"],
                                                class_weight="balanced", random_state=0)).fit(X, y)
    rules = export_text(tree.named_steps["decisiontreeclassifier"],
                        feature_names=[f for f in feats], max_depth=t["max_depth"])
    return {"wallet": wallet, "n": len(e), "fidelity": fid, "null": null,
            "groups": grp, "uni": uni_auc(e, base, feats), "rules": rules}


def _self_test() -> None:
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    print("building full entry-state matrix + universe baseline ...")
    frames = features.load_snapshot_frames()
    enr = features.token_enrichment()
    ent = features.entry_state_full(data.load_events(), frames)
    ent = ent[ent["covered"]]
    feats = [f for f in model_features() if f in ent.columns]
    base = universe_baseline(frames, enr)
    feats = [f for f in feats if f in base.columns]
    groups = features.feature_groups()
    print(f"  entries={len(ent):,}  baseline={len(base):,}  features={len(feats)}\n")

    fid_rows, uni_rows = [], []
    print(f"  {'wallet':10} {'n':>5} {'FID':>5} {'null':>5} | "
          + " ".join(f"{g[:6]:>6}" for g in groups))
    for w in sorted(ent["wallet"].unique()):
        r = analyse(w, ent, base, feats)
        gtxt = " ".join(f"{r['groups'][g]:>6.2f}" if r['groups'][g] == r['groups'][g] else "   nan"
                        for g in groups)
        print(f"  {w:10} {r['n']:>5} {r['fidelity']:>5.2f} {r['null']:>5.2f} | {gtxt}")
        fid_rows.append({"wallet": w, "n": r["n"], "fidelity": r["fidelity"], "null": r["null"],
                         **{f"grp_{g}": r["groups"][g] for g in groups}})
        for f, a, em, bm in r["uni"][:P("entry.report_top_n")]:
            uni_rows.append({"wallet": w, "feature": f, "auc": a, "entry_med": em, "base_med": bm})

    print("\n=== top discriminators across ALL feature families (entry vs universe) ===")
    udf = pd.DataFrame(uni_rows)
    for w in sorted(ent["wallet"].unique()):
        print(f"\n  --- {w} ---")
        for _, x in udf[udf.wallet == w].iterrows():
            d = "HIGHER" if x.entry_med > x.base_med else "lower"
            print(f"     {x.feature:24} AUC {x.auc:.2f}  entry {x.entry_med:>11.4g} vs univ {x.base_med:>11.4g} -> {d}")

    pd.DataFrame(fid_rows).round(4).to_csv(config.OUTPUT_DIR / "c2_fidelity.csv", index=False)
    udf.to_csv(config.OUTPUT_DIR / "c2_discriminators.csv", index=False)
    print("\n  FID = OOS entry-vs-universe AUC (GroupKFold-by-token) | null must be ~0.50")
    print("  group cols = fidelity from THAT family alone -> which data distinguishes a buy")
    print("\nC2 OK - entry decoded on the FULL inventory; written to analysis/outputs/.")


if __name__ == "__main__":
    _self_test()
