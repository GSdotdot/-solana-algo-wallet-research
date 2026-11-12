"""C3 — exit rules, decoded as a discrete-time survival / hazard problem.

A first version of this step was descriptive: crude cut-offs on hold dispersion and
take-profit spread. This rebuild holds the C2 rigor bar (token-grouped validation,
shuffle nulls) and adds the survival machinery the exit question actually needs.

Three layers:
  L0  survival   : Kaplan-Meier hold curves WITH right-censoring (positions still open
                   when the data ends) + partial-exit / FLIP accounting. Removes the
                   survivorship bias of looking at closed round-trips only.
  L1  hazard     : per in-hold observation, classify exit-now vs hold from a CAUSAL feature
                   vector (running PnL, drawdown-from-peak, momentum, time-in-trade, on-chain
                   deltas). Univariate AUC + readable tree + GBM fidelity ceiling, C2-style.
  L2  adjudicate : which family of signals best times the exit: time in trade alone, the
                   price path since entry (running PnL, drawdown from peak, run-up,
                   momentum), or on-chain changes (volume, sell pressure, holders /
                   liquidity, trader types, structure), scored by the OUT-OF-SAMPLE lift
                   of each family, within each hold, over price alone.
  + robustness   : GroupKFold-by-token (no token leakage), within-hold label-shuffle null,
                   temporal holdout (train early positions, test late ones), and a
                   partial-exit sensitivity check.

PATH SOURCE (hybrid, best-available per position):
  * Fast-scalper ("B") wallets, in the period when 1-second price bars were being
    stored: the bars where they are denser, else the snapshot price. Earlier B
    positions and all other wallets: snapshot price.
  * On-chain state always from snapshots (forward-filled). Because the underlying
    fields refresh slowly, we use ABSOLUTE state + since-entry change only, never
    snapshot-to-snapshot velocities (degenerate, see config).
  * Running PnL anchored to the ACTUAL execution price (volume_usd / amount_token), not the
    lagged on-chain `price` and not the SOL-denominated price (different scale).

All lags, windows, sample minimums, model settings and verdict cut-offs are read
from config (`analysis.exit.*`) and redacted in the public copy.

Run `python -m analysis.exit`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config, data, features

warnings.filterwarnings("ignore")

P = config.P

OPEN_LABELS = ("ENTRY", "FLIP")

# Snapshot columns we carry as on-chain STATE (absolute). They refresh slowly, so they are
# safe as levels + since-entry deltas, NOT as snapshot-to-snapshot velocities.
SNAP_COLS = [
    "token_address", "timestamp", "price", "liquidity", "market_cap", "holder",
    "progress_percent", "top10_holder_percentage", "buy_sell_ratio_1m", "buy_sell_ratio_5m",
    "sniper_held_percentage", "bundler_held_percentage", "volume_5m_usd", "buy_5m", "sell_5m",
    "volume_1m_usd", "volume_buy_1m_usd", "volume_sell_1m_usd",
    "volume_buy_5m_usd", "volume_sell_5m_usd",
]
STATE_LEVEL = ["liquidity", "market_cap", "holder", "progress_percent",
               "top10_holder_percentage", "buy_sell_ratio_1m", "buy_sell_ratio_5m",
               "sniper_held_percentage", "bundler_held_percentage", "volume_5m_usd"]
# on-chain DYNAMICS — cadence-matched short-window change/pressure (the exit-signal test).
# Each lag is >= that field's measured refresh cadence (config.CADENCE_SEC), so these are
# REAL movements, not the degenerate live velocities config.DEGENERATE warns about.
# Lags (seconds) come from analysis.exit.dynamics_lags_sec; tuned privately.
STATE_DYN = ["sell_frac_1m", "sell_frac_5m", "log_vol_1m", "net_flow_1m_norm", "vol_accel",
             "holder_d_short", "holder_d_long", "top10_d", "liq_d_pct", "sniper_d",
             "bundler_d", "bar_vol_fast", "bar_vol_burst"]
# Non-monotonic price-path features (exclude t_in_trade/mfe/mae — mechanically extreme at the
# terminal obs, so they trivially mark it and corrupt a within-hold trigger test).
NONMONO_PATH = ["run_pnl", "runup_from_trough", "dd_from_peak", "mom_fast", "mom_slow", "frac_of_mfe"]

MIN_OBS = P("exit.min_obs")    # in-hold observations a position needs to model its timing; tuned privately
OBS_CAP = P("exit.obs_cap")    # cap obs/position so a few long holds can't dominate the panel; tuned privately


# --- bulk loaders (one query each; slice in memory) ------------------------------
def load_snap_frames() -> dict[str, pd.DataFrame]:
    con = data._conn()
    df = pd.read_sql(f"SELECT {','.join(SNAP_COLS)} FROM token_snapshots ORDER BY token_address, timestamp", con)
    con.close()
    for c in df.columns:
        if c != "token_address":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return {t: g.reset_index(drop=True) for t, g in df.groupby("token_address", sort=False)}


def load_tick_frames() -> dict[str, pd.DataFrame]:
    con = data._conn()
    df = pd.read_sql("SELECT token_address, unix_time, ohlcv_close, volume_usd FROM price_ticks "
                     "WHERE ohlcv_close > 0 ORDER BY token_address, unix_time", con)
    con.close()
    return {t: g.reset_index(drop=True) for t, g in df.groupby("token_address", sort=False)}


def tick_era_start() -> int:
    con = data._conn()
    v = con.execute("SELECT MIN(unix_time) FROM price_ticks").fetchone()[0]
    con.close()
    return int(v)


def data_end() -> int:
    con = data._conn()
    v = con.execute("SELECT MAX(timestamp) FROM token_snapshots").fetchone()[0]
    con.close()
    return int(v)


# --- execution prices (anchor for running PnL) -----------------------------------
def exec_price_maps(events: pd.DataFrame) -> tuple[dict, dict]:
    """(entry_exec, exit_exec) keyed (wallet, token, timestamp) -> USD/token execution price
    = volume_usd / amount_token. 100% coverage, matches the snapshot/tick price scale."""
    e = events[["wallet", "token_address", "timestamp", "label", "volume_usd", "amount_token"]].copy()
    e = e[(e["amount_token"] > 0) & (e["volume_usd"] > 0)]
    e["exec_usd"] = e["volume_usd"] / e["amount_token"]
    op = e[e["label"].isin(OPEN_LABELS)]
    ex = e[e["label"] == "EXIT"]
    k = lambda g: {(w, t, int(ts)): v for w, t, ts, v in
                   zip(g["wallet"], g["token_address"], g["timestamp"], g["exec_usd"])}
    return k(op), k(ex)


# --- per-position path + features ------------------------------------------------
def _causal_momentum(ts: np.ndarray, px: np.ndarray, lag: int) -> np.ndarray:
    """price[t] / price[<=t-lag seconds] - 1, causal (searchsorted on sorted ts)."""
    j = np.searchsorted(ts, ts - lag, side="right") - 1
    out = np.full(len(px), np.nan)
    ok = j >= 0
    out[ok] = px[ok] / px[j[ok]] - 1.0
    return out


def build_paths(pos: pd.DataFrame, snaps: dict, ticks: dict, entry_exec: dict, exit_exec: dict,
                tes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (exit_summary, hazard_panel).

    exit_summary : one row per modellable position — terminal path features + realized
                   outcome + era + path_source + n_obs + resolution.
    hazard_panel : one row per in-hold observation — causal features + label(exit-now) +
                   group keys (wallet, token, position_id, era). The L1/L2 training matrix.
    """
    sum_rows, panel_parts = [], []
    p = pos[pos["tracked"] & pos["entry_time"].notna() & pos["exit_time"].notna()].copy()
    for pid, r in enumerate(p.to_dict("records")):
        w, tok = r["wallet"], r["token_address"]
        e_t, x_t = int(r["entry_time"]), int(r["exit_time"])
        sf = snaps.get(tok)
        if sf is None:
            continue
        swin = sf[(sf["timestamp"] >= e_t) & (sf["timestamp"] <= x_t)]
        is_b = w in config.B_WALLET_LABELS
        tick_era = e_t >= tes
        # choose denser price source
        twin = None
        if is_b and tick_era and tok in ticks:
            tf = ticks[tok]
            twin = tf[(tf["unix_time"] >= e_t) & (tf["unix_time"] <= x_t)]
            if len(twin) <= len(swin):
                twin = None
        if twin is not None and len(twin) >= MIN_OBS:
            ts = twin["unix_time"].to_numpy(float)
            px = twin["ohlcv_close"].to_numpy(float)
            tvol = twin["volume_usd"].to_numpy(float)        # per-bar $ volume (1s bars)
            source = "tick"
        elif len(swin) >= MIN_OBS:
            ts = swin["timestamp"].to_numpy(float)
            px = swin["price"].to_numpy(float)
            tvol = None
            source = "snapshot"
        else:
            continue
        # de-dup / sort by time, drop non-positive prices
        order = np.argsort(ts, kind="stable")
        ts, px = ts[order], px[order]
        good = px > 0
        ts, px = ts[good], px[good]
        if tvol is not None:
            tvol = tvol[order][good]
        if len(ts) < MIN_OBS:
            continue

        ee = entry_exec.get((w, tok, e_t))
        if not ee or ee <= 0:
            ee = px[0]                                  # fallback: first path price
        # --- causal path features ---
        cummax = np.maximum.accumulate(px)
        cummin = np.minimum.accumulate(px)
        run_pnl = px / ee - 1.0
        mfe = cummax / ee - 1.0
        mae = cummin / ee - 1.0
        dd_from_peak = np.where(cummax > 0, (cummax - px) / cummax, 0.0)
        runup = np.where(cummin > 0, (px - cummin) / cummin, 0.0)
        frac_of_mfe = np.where(mfe > 1e-9, run_pnl / mfe, np.nan)
        t_in = ts - e_t
        mom_lag = P("exit.momentum_lags_sec")              # {fast, slow}; tuned privately
        mom_fast = _causal_momentum(ts, px, mom_lag["fast"])
        mom_slow = _causal_momentum(ts, px, mom_lag["slow"])

        n = len(ts)
        panel = pd.DataFrame({
            "wallet": w, "token_address": tok, "position_id": pid,
            "era": "tick" if tick_era else "pre", "source": source, "entry_time": e_t,
            "t_in_trade": t_in, "run_pnl": run_pnl, "mfe": mfe, "mae": mae,
            "dd_from_peak": dd_from_peak, "runup_from_trough": runup, "frac_of_mfe": frac_of_mfe,
            "mom_fast": mom_fast, "mom_slow": mom_slow,
        })
        # --- on-chain STATE + DYNAMICS forward-filled from snapshots onto path times ---
        sft = sf[sf["timestamp"] <= x_t]
        if len(sft):
            sts = sft["timestamp"].to_numpy()
            arr = {c: pd.to_numeric(sft[c], errors="coerce").to_numpy() for c in SNAP_COLS
                   if c not in ("token_address", "timestamp")}

            def at(c, lb=0):                                  # causal forward-fill at time t-lb
                j = np.clip(np.searchsorted(sts, ts - lb, side="right") - 1, 0, len(sts) - 1)
                return arr[c][j]

            ent_idx = max(np.searchsorted(sts, e_t, side="right") - 1, 0)
            for c in STATE_LEVEL:                             # absolute levels + since-entry change
                v = at(c); panel[c] = v
                base = arr[c][ent_idx]
                if c == "holder":
                    panel["holder_chg"] = v - base
                elif c in ("liquidity", "market_cap"):
                    panel[f"{c}_chg_pct"] = np.where(base > 0, v / base - 1.0, np.nan)
            vol1, vsell1, vbuy1 = at("volume_1m_usd"), at("volume_sell_1m_usd"), at("volume_buy_1m_usd")
            vsell5, vbuy5, liq = at("volume_sell_5m_usd"), at("volume_buy_5m_usd"), at("liquidity")
            panel["sell_frac_1m"] = vsell1 / (vol1 + 1e-9)                       # sell pressure now
            panel["sell_frac_5m"] = vsell5 / (vbuy5 + vsell5 + 1e-9)
            panel["log_vol_1m"] = np.log1p(vol1)
            panel["net_flow_1m_norm"] = (vbuy1 - vsell1) / (liq + 1e-9)          # net $ flow / liq
            lag = P("exit.dynamics_lags_sec")                 # per-field lags; tuned privately
            panel["vol_accel"] = vol1 - at("volume_1m_usd", lag["vol_accel"])    # volume burst
            panel["holder_d_short"] = at("holder") - at("holder", lag["holder_short"])  # holders leaving?
            panel["holder_d_long"] = at("holder") - at("holder", lag["holder_long"])
            panel["top10_d"] = at("top10_holder_percentage") - at("top10_holder_percentage", lag["top10"])
            panel["liq_d_pct"] = liq / (at("liquidity", lag["liquidity"]) + 1e-9) - 1.0   # liquidity pull?
            panel["sniper_d"] = at("sniper_held_percentage") - at("sniper_held_percentage", lag["trader_class"])
            panel["bundler_d"] = at("bundler_held_percentage") - at("bundler_held_percentage", lag["trader_class"])
        if tvol is not None:                                  # 1s-bar volume bursts (B-wallets)
            cs = np.concatenate([[0.0], np.cumsum(tvol)])
            rsum = lambda win: cs[np.arange(len(ts)) + 1] - cs[np.searchsorted(ts, ts - win, side="left")]
            tw = P("exit.bar_volume_windows_sec")             # {fast, slow}; tuned privately
            v_fast, v_slow = rsum(tw["fast"]), rsum(tw["slow"])
            panel["bar_vol_fast"] = np.log1p(v_fast)
            panel["bar_vol_burst"] = v_fast / (v_slow / (tw["slow"] / tw["fast"]) + 1e-9)
        else:
            panel["bar_vol_fast"] = np.nan
            panel["bar_vol_burst"] = np.nan
        panel["label"] = 0
        panel.iloc[-1, panel.columns.get_loc("label")] = 1     # terminal obs = exit-now
        if len(panel) > OBS_CAP:                               # bound any single position's weight
            keep = np.unique(np.r_[np.linspace(0, len(panel) - 2, OBS_CAP - 1).round().astype(int),
                                   len(panel) - 1])            # evenly-spaced holds + always the terminal
            panel = panel.iloc[keep].reset_index(drop=True)
        panel_parts.append(panel)

        realized = (r["net_pnl_usd"] / r["size_usd"] * 100) if r.get("size_usd") else np.nan
        sum_rows.append({
            "wallet": w, "token_address": tok, "position_id": pid, "era": panel["era"].iloc[0],
            "source": source, "n_obs": n, "res_sec": float(np.median(np.diff(ts))) if n > 1 else np.nan,
            "hold_sec": x_t - e_t, "entry_time": e_t, "size_usd": r.get("size_usd"),
            "mfe_pct": mfe[-1] * 100, "mae_pct": mae[-1] * 100,
            "exit_run_pnl_pct": run_pnl[-1] * 100, "exit_dd_from_peak_pct": dd_from_peak[-1] * 100,
            "exit_mom_fast_pct": mom_fast[-1] * 100, "exit_frac_of_mfe": frac_of_mfe[-1],
            "realized_pct": realized, "win": realized > 0 if realized == realized else np.nan,
        })
    exit_summary = pd.DataFrame(sum_rows)
    hazard_panel = pd.concat(panel_parts, ignore_index=True) if panel_parts else pd.DataFrame()
    return exit_summary, hazard_panel


# --- L0 survival: Kaplan-Meier with right-censoring ------------------------------
def km_estimate(durations: np.ndarray, observed: np.ndarray) -> pd.DataFrame:
    """Manual KM. observed=1 exit seen, 0 right-censored (open at freeze)."""
    d = np.asarray(durations, float); o = np.asarray(observed, float)
    keep = d >= 0
    d, o = d[keep], o[keep]
    order = np.argsort(d); d, o = d[order], o[order]
    surv, rows = 1.0, []
    for t in np.unique(d):
        at_risk = (d >= t).sum()
        events = o[d == t].sum()
        if at_risk > 0:
            surv *= 1 - events / at_risk
        rows.append({"t": t, "S": surv, "at_risk": at_risk, "events": events})
    return pd.DataFrame(rows)


def km_median(km: pd.DataFrame) -> float:
    below = km[km["S"] <= 0.5]
    return float(below["t"].iloc[0]) if len(below) else np.nan


def censored_positions(events: pd.DataFrame, end_t: int) -> pd.DataFrame:
    """Opens with no matching EXIT at freeze -> right-censored, duration = freeze - entry."""
    rows = []
    ev = events.sort_values(["wallet", "token_address", "timestamp", "event_id"])
    for (w, tok), g in ev.groupby(["wallet", "token_address"], sort=False):
        stack = []
        for e in g.to_dict("records"):
            if e["label"] in OPEN_LABELS:
                stack.append(e["timestamp"])
            elif e["label"] == "EXIT" and stack:
                stack.pop()
        for ts in stack:
            rows.append({"wallet": w, "token_address": tok, "hold_sec": end_t - ts, "observed": 0})
    return pd.DataFrame(rows)


def partial_exit_accounting(events: pd.DataFrame) -> pd.DataFrame:
    """What the clean-round-trip view DROPS: PARTIAL_EXIT / ADD / FLIP share per wallet."""
    rows = []
    for w, g in events.groupby("wallet"):
        vc = g["label"].value_counts()
        opens = vc.get("ENTRY", 0) + vc.get("FLIP", 0)
        rows.append({
            "wallet": w, "ENTRY": vc.get("ENTRY", 0), "FLIP": vc.get("FLIP", 0),
            "ADD": vc.get("ADD", 0), "PARTIAL_EXIT": vc.get("PARTIAL_EXIT", 0),
            "EXIT": vc.get("EXIT", 0),
            "partial_exit_rate": vc.get("PARTIAL_EXIT", 0) / max(opens, 1),
            "flip_rate": vc.get("FLIP", 0) / max(opens, 1),
        })
    return pd.DataFrame(rows).set_index("wallet")


# --- L1/L2 hazard modeling -------------------------------------------------------
TIME_FEATS = ["t_in_trade"]
PATH_FEATS = ["t_in_trade", "run_pnl", "mfe", "mae", "dd_from_peak", "runup_from_trough",
              "frac_of_mfe", "mom_fast", "mom_slow"]


def feature_sets(panel: pd.DataFrame) -> tuple[list, list, list]:
    """(time-only, +path, +path+state) — each filtered to columns present in `panel`."""
    state = [c for c in panel.columns
             if c in STATE_LEVEL or c.endswith("_chg") or c.endswith("_chg_pct")]
    dyn = [c for c in STATE_DYN if c in panel.columns]
    path = [f for f in PATH_FEATS if f in panel.columns]
    full = path + [c for c in state + dyn if c not in path]
    return [f for f in TIME_FEATS if f in panel.columns], path, full


def _hgb() -> HistGradientBoostingClassifier:
    m = P("exit.model")                     # depth / iterations / learning rate: tuned privately
    return HistGradientBoostingClassifier(max_depth=m["max_depth"], max_iter=m["max_iter"],
                                          learning_rate=m["learning_rate"],
                                          class_weight="balanced", random_state=0)


def cv_auc(panel: pd.DataFrame, feats: list[str], n_splits: int | None = None) -> float:
    """Out-of-sample AUC, GroupKFold by TOKEN (a token's obs never span train/test: the
    key anti-leakage fix vs a random split)."""
    if n_splits is None:
        n_splits = P("cv_folds")
    min_class = P("exit.min_class_count")
    y = panel["label"].to_numpy()
    g = panel["token_address"]
    if g.nunique() < n_splits or y.sum() < min_class or (1 - y).sum() < min_class:
        return np.nan
    cv = GroupKFold(n_splits=min(n_splits, g.nunique()))
    return float(cross_val_score(_hgb(), panel[feats], y, groups=g, cv=cv, scoring="roc_auc").mean())


def within_position_auc(panel: pd.DataFrame, feats: list[str]) -> tuple[float, int]:
    """Can we pick the EXIT observation from WITHIN each hold? Out-of-fold (GroupKFold-by-token)
    exit-probabilities, then per position rank the true exit obs against its hold siblings;
    average. Immune to between-position composition AND to absolute levels (~constant within a
    hold). 0.50 = no within-hold trigger; >0.50 = a real within-hold exit signal. THE metric
    for 'does a price/on-chain signal time the exit'."""
    folds = P("cv_folds")
    y = panel["label"].to_numpy()
    g = panel["token_address"]
    if g.nunique() < folds or y.sum() < P("exit.min_class_count"):
        return (np.nan, 0)
    p = cross_val_predict(_hgb(), panel[feats], y, groups=g, cv=GroupKFold(min(folds, g.nunique())),
                          method="predict_proba")[:, 1]
    df = pd.DataFrame({"pid": panel["position_id"].to_numpy(), "y": y, "p": p})
    aucs = []
    for _, q in df.groupby("pid"):
        ex, ho = q["p"][q["y"] == 1].to_numpy(), q["p"][q["y"] == 0].to_numpy()
        if len(ex) and len(ho):
            aucs.append(np.mean([(ho < e).mean() + 0.5 * (ho == e).mean() for e in ex]))
    return (float(np.mean(aucs)), len(aucs))


def within_position_null(panel: pd.DataFrame, feats: list[str], seed: int = 0) -> float:
    """Within-position AUC with the exit label SHUFFLED within each hold. Must collapse to
    ~0.50 — if it doesn't, the within-hold lift is overfit, not a real trigger."""
    rng = np.random.default_rng(seed)
    lab = np.zeros(len(panel), dtype=int)
    for idx in panel.groupby("position_id").indices.values():
        lab[idx[rng.integers(len(idx))]] = 1
    return within_position_auc(panel.assign(label=lab), feats)[0]


def uni_auc(panel: pd.DataFrame, feats: list[str]) -> list[tuple]:
    """Single-feature AUC (exit-obs vs hold-obs) + median at exit vs in-hold. Descriptive /
    directional (full-panel, correlated rows) — the CV fidelity is the rigorous number."""
    y = panel["label"].to_numpy()
    ex, hold = panel[panel["label"] == 1], panel[panel["label"] == 0]
    out = []
    min_obs = P("exit.univariate_min_obs")
    for f in feats:
        x = panel[f]; m = x.notna().to_numpy()
        if m.sum() < min_obs or y[m].sum() in (0, m.sum()):
            continue
        try:
            a = roc_auc_score(y[m], x[m].to_numpy())
        except ValueError:
            continue
        out.append((f, a, ex[f].median(), hold[f].median()))
    out.sort(key=lambda r: -abs(r[1] - 0.5))
    return out


def rule_tree(panel: pd.DataFrame, feats: list[str], depth: int | None = None) -> str:
    t = P("exit.rule_tree")                 # depth / leaf sizing: tuned privately
    if depth is None:
        depth = t["max_depth"]
    feats = [f for f in feats if panel[f].notna().any()]   # SimpleImputer drops all-NaN cols
    leaf = max(t["min_leaf_floor"], len(panel) // t["leaf_panel_divisor"])
    tree = make_pipeline(SimpleImputer(strategy="median"),
                         DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                                class_weight="balanced", random_state=0))
    tree.fit(panel[feats], panel["label"])
    return export_text(tree.named_steps["decisiontreeclassifier"], feature_names=feats, max_depth=depth)


def model_segment(panel: pd.DataFrame) -> dict:
    """Fidelity decomposition for one (wallet[/era]) segment."""
    t, path, full = feature_sets(panel)
    return {
        "n_pos": panel["position_id"].nunique(), "n_obs": len(panel),
        "n_tokens": panel["token_address"].nunique(),
        "auc_time": cv_auc(panel, t), "auc_path": cv_auc(panel, path),
        "auc_full": cv_auc(panel, full), "uni": uni_auc(panel, full), "full_feats": full,
    }


# --- L3 robustness battery -------------------------------------------------------
def temporal_holdout(panel: pd.DataFrame, feats: list[str], frac: float | None = None) -> tuple[float, float]:
    """Train on the EARLY positions, test on the LATE ones (out-of-time). A big train>>test
    gap = the rule is memorized/regime-bound, not stable. Split is position-level."""
    th = P("exit.temporal_holdout")         # {train_frac, min_events}; tuned privately
    if frac is None:
        frac = th["train_frac"]
    pt = panel.groupby("position_id")["entry_time"].first().sort_values()
    cut = pt.quantile(frac)
    tr = panel[panel["position_id"].isin(pt[pt <= cut].index)]
    te = panel[panel["position_id"].isin(pt[pt > cut].index)]
    if tr["label"].sum() < th["min_events"] or te["label"].sum() < th["min_events"]:
        return (np.nan, np.nan)
    clf = _hgb().fit(tr[feats], tr["label"])
    tr_auc = roc_auc_score(tr["label"], clf.predict_proba(tr[feats])[:, 1])
    te_auc = roc_auc_score(te["label"], clf.predict_proba(te[feats])[:, 1])
    return float(tr_auc), float(te_auc)


def position_balanced_uni(panel: pd.DataFrame, feats: list[str], seed: int = 0) -> list[tuple]:
    """Univariate AUC of terminal-obs vs ONE random hold-obs per position (each position
    weighted equally) -> neutralises the obs-domination that distorts the raw-panel medians
    (matters most when a wallet's few long holds would otherwise dominate the raw panel)."""
    rng = np.random.default_rng(seed)
    ex = panel[panel["label"] == 1]
    hold = panel[panel["label"] == 0].sample(frac=1.0, random_state=seed).drop_duplicates("position_id")
    bal = pd.concat([ex.assign(label=1), hold.assign(label=0)], ignore_index=True)
    return uni_auc(bal, feats)


# --- L2 partial-exit sensitivity -------------------------------------------------
def partial_exit_map(events: pd.DataFrame) -> dict:
    pe = events[events["label"] == "PARTIAL_EXIT"]
    m = {}
    for w, t, ts in zip(pe["wallet"], pe["token_address"], pe["timestamp"]):
        m.setdefault((w, t), []).append(int(ts))
    return m


def add_partial_positives(panel: pd.DataFrame, pmap: dict) -> pd.DataFrame:
    """Mark the obs nearest each PARTIAL_EXIT as exit-now too (scale-outs treated as exit
    decisions) -> does the decoded rule change when we stop dropping them?"""
    pn = panel.reset_index(drop=True).copy()           # positional index so .indices aligns
    pn["obs_time"] = pn["entry_time"] + pn["t_in_trade"]
    lab = pn["label"].to_numpy().copy()
    ot_all = pn["obs_time"].to_numpy()
    for (w, tok), idx in pn.groupby(["wallet", "token_address"]).indices.items():
        pts = pmap.get((w, tok))
        if not pts:
            continue
        ot = ot_all[idx]; lo, hi = ot.min(), ot.max()
        for t in pts:
            if lo <= t <= hi:
                lab[idx[np.abs(ot - t).argmin()]] = 1
    pn["label"] = lab
    return pn.drop(columns="obs_time")


# --- full-inventory on-chain attach (doctrine: exit coverage == entry coverage) -----
META_COLS = {"wallet", "token_address", "position_id", "era", "source", "entry_time", "label",
             "t_in_trade", "run_pnl", "mfe", "mae", "dd_from_peak", "runup_from_trough",
             "frac_of_mfe", "mom_fast", "mom_slow"}


def onchain_cols(panel: pd.DataFrame) -> list[str]:
    """Every NON-price-path column = the on-chain inventory attached to the hold path."""
    return [c for c in panel.columns if c not in META_COLS]


def attach_onchain(panel: pd.DataFrame, frames: dict) -> pd.DataFrame:
    """Attach the FULL features.snapshot_features ladder at each in-hold observation (batched
    per token; on-chain state is a pure function of (token, time)). De-dups against the
    validated price-path/slice columns, so the panel is preserved and the complete inventory —
    all-timeframe volume/sell ladder, insider/dev classifications, supply/reserve dynamics — is
    added on top."""
    parts = []
    for tok, g in panel.groupby("token_address", sort=False):
        fr = frames.get(tok)
        if fr is None:
            parts.append(pd.DataFrame(index=g.index)); continue
        f = features.snapshot_features(fr, (g["entry_time"] + g["t_in_trade"]).to_numpy())
        f.index = g.index
        parts.append(f)
    onc = pd.concat(parts).reindex(panel.index)
    # de-dup: the panel's own holder deltas (holder_d_short / holder_d_long) are the same
    # quantities as the factory's holder delta columns, so the factory copies are dropped
    dup = set(config.delta_feature_names(["holder"]))
    onc = onc[[c for c in onc.columns if c not in panel.columns and c not in dup]]
    return pd.concat([panel, onc], axis=1)


def onchain_groups(panel: pd.DataFrame) -> dict[str, list]:
    """Bucket on-chain columns into families for within-hold attribution (now incl. the
    previously-unused insider/dev classes + the full-timeframe volume/sell ladder)."""
    grp = {"volume": [], "sellpress": [], "holder_liq": [], "classif": [], "structural": []}
    for c in onchain_cols(panel):
        if any(k in c for k in ("sell_frac", "net_flow", "trade_imbalance", "buy_sell_ratio")):
            grp["sellpress"].append(c)
        elif any(k in c for k in ("sniper", "bundler", "insider", "dev")):
            grp["classif"].append(c)
        elif any(k in c for k in ("holder", "liquidity", "top10")):
            grp["holder_liq"].append(c)
        elif (any(k in c for k in ("log_vol", "n_trades", "vol_accel", "bar_vol"))
              or c.startswith(("buy_", "sell_", "volume_"))):
            grp["volume"].append(c)
        elif any(k in c for k in ("market_cap", "fdv", "price", "progress", "supply", "reserve", "graduated")):
            grp["structural"].append(c)
    return grp


# --- self-test: STAGE 1 (data + survival) ----------------------------------------
def _self_test() -> None:
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    ev = data.load_events()
    pos = data.reconstruct_positions(ev)
    tes, end_t = tick_era_start(), data_end()
    print(f"1s-bar era = last {(end_t-tes)/3600:.1f}h of the data")

    print("\nloading FULL snapshot + tick frames ...")
    snaps, ticks = features.load_snapshot_frames(), load_tick_frames()   # full inventory frames
    entry_exec, exit_exec = exec_price_maps(ev)
    print(f"  snapshot tokens={len(snaps):,}  tick tokens={len(ticks):,}")

    summ, panel = build_paths(pos, snaps, ticks, entry_exec, exit_exec, tes)
    _np0 = panel.shape[1]
    panel = attach_onchain(panel, snaps)                                 # full on-chain ladder at each obs
    print(f"  attached full on-chain inventory: {_np0} -> {panel.shape[1]} cols/obs")
    print(f"\n=== PANEL BUILT ===  positions modelled={len(summ):,}  obs={len(panel):,}")
    print("  per-wallet: modelled / source mix / median resolution / median obs")
    for w, g in summ.groupby("wallet"):
        src = g["source"].value_counts(normalize=True)
        print(f"    {w:10} n={len(g):>5}  tick={src.get('tick',0)*100:>3.0f}% snap={src.get('snapshot',0)*100:>3.0f}%"
              f"  res~{g['res_sec'].median():>4.1f}s  obs~{g['n_obs'].median():>4.0f}  "
              f"era[pre/tick]={int((g.era=='pre').sum())}/{int((g.era=='tick').sum())}")
    # coverage honesty: how many tracked closed positions did NOT make the panel
    closed = pos[pos["tracked"]].dropna(subset=["entry_time", "exit_time"])
    print(f"\n  panel coverage: {len(summ):,}/{len(closed):,} closed-tracked positions "
          f"({len(summ)/len(closed)*100:.0f}%); below-{MIN_OBS}-obs excluded (reported in survival).")
    for w, g in closed.groupby("wallet"):
        nin = (summ["wallet"] == w).sum() if len(summ) else 0
        print(f"    {w:10} modelled {nin}/{len(g)} ({nin/len(g)*100:>3.0f}%)")

    print("\n=== L0  SURVIVAL (Kaplan-Meier, right-censored) ===")
    cens = censored_positions(ev, end_t)
    clo = pos[pos["tracked"]].dropna(subset=["hold_duration_sec"])
    for w in sorted(pos["wallet"].unique()):
        dur = clo.loc[clo["wallet"] == w, "hold_duration_sec"].to_numpy()
        obs = np.ones(len(dur))
        cw = cens[cens["wallet"] == w]
        d = np.r_[dur, cw["hold_sec"].to_numpy()]
        o = np.r_[obs, np.zeros(len(cw))]
        km = km_estimate(d, o)
        hz = P("exit.survival_report_horizon_sec")     # horizon at which survival is reported
        print(f"    {w:10} KM-median={km_median(km):>6.0f}s  n_closed={len(dur):>5} censored={len(cw):>3} "
              f"({len(cw)/(len(dur)+len(cw))*100:>3.1f}%)  S@horizon={km[km.t<=hz]['S'].iloc[-1] if (km.t<=hz).any() else 1:.2f}")

    pea = partial_exit_accounting(ev)
    print("\n  partial-exit / FLIP share (dropped by clean round-trips - sensitivity in L2):")
    print(pea[["partial_exit_rate", "flip_rate"]].round(2).to_string())
    summ.to_csv(config.OUTPUT_DIR / "c3_exit_summary.csv", index=False)

    # === L1  EXIT-TIMING HAZARD MODEL (CV AUC, GroupKFold-by-token) ===
    print("\n=== L1  EXIT-TIMING HAZARD: out-of-sample AUC (GroupKFold-by-token) ===")
    print(f"  {'wallet':10} {'seg':5} {'npos':>5} {'nobs':>7} {'tok':>4} | "
          f"{'time':>5} {'+path':>6} {'+state':>6} | {'liftP':>6} {'liftS':>6}")
    fid_rows = []
    segments = []                                   # (wallet, seg, panel)
    for w in sorted(panel["wallet"].unique()):
        wp = panel[panel["wallet"] == w]
        segments.append((w, "all", wp))
        if w in config.B_WALLET_LABELS:             # parallel pre-tick vs tick-era analysis
            for era in ("pre", "tick"):
                ep = wp[wp["era"] == era]
                if ep["position_id"].nunique() >= P("exit.era_min_positions"):
                    segments.append((w, era, ep))
    seg_models = {}
    for w, seg, sp in segments:
        m = model_segment(sp)
        seg_models[(w, seg)] = m
        lp = (m["auc_path"] - m["auc_time"]) if (m["auc_path"] == m["auc_path"] and m["auc_time"] == m["auc_time"]) else np.nan
        ls = (m["auc_full"] - m["auc_path"]) if (m["auc_full"] == m["auc_full"] and m["auc_path"] == m["auc_path"]) else np.nan
        print(f"  {w:10} {seg:5} {m['n_pos']:>5} {m['n_obs']:>7} {m['n_tokens']:>4} | "
              f"{m['auc_time']:>5.2f} {m['auc_path']:>6.2f} {m['auc_full']:>6.2f} | {lp:>+6.2f} {ls:>+6.2f}")
        fid_rows.append({"wallet": w, "seg": seg, "n_pos": m["n_pos"], "n_obs": m["n_obs"],
                         "n_tokens": m["n_tokens"], "auc_time": m["auc_time"], "auc_path": m["auc_path"],
                         "auc_full": m["auc_full"], "lift_path": lp, "lift_state": ls})
    pd.DataFrame(fid_rows).round(4).to_csv(config.OUTPUT_DIR / "c3_hazard_fidelity.csv", index=False)

    print("\n=== L1  top exit discriminators + readable rule (per wallet, 'all' segment) ===")
    uni_rows = []
    for w in sorted(panel["wallet"].unique()):
        m = seg_models[(w, "all")]
        print(f"\n  --- {w}  (fidelity {m['auc_full']:.2f}, time-only {m['auc_time']:.2f}) ---")
        for f, a, em, hm in m["uni"][:P("exit.report_top_n")]:
            d = "HIGHER" if em > hm else "lower"
            print(f"     {f:20} AUC {a:.2f}  exit {em:>11.4g} vs hold {hm:>11.4g}  -> {d} at exit")
            uni_rows.append({"wallet": w, "feature": f, "auc": a, "exit_med": em, "hold_med": hm})
        wp = panel[panel["wallet"] == w]
        print("     rule (shallow surrogate tree, exit=1):")
        print("      " + rule_tree(wp, m["full_feats"]).replace("\n", "\n      ").rstrip())
    pd.DataFrame(uni_rows).to_csv(config.OUTPUT_DIR / "c3_exit_discriminators.csv", index=False)

    # === L2b  WITHIN-HOLD DECODABILITY + ON-CHAIN ATTRIBUTION (artifact-immune) ===
    km_meds = {}
    cens = censored_positions(ev, end_t)
    clo = pos[pos["tracked"]].dropna(subset=["hold_duration_sec"])
    for w in sorted(panel["wallet"].unique()):
        dur = clo.loc[clo["wallet"] == w, "hold_duration_sec"].to_numpy()
        cw = cens[cens["wallet"] == w]
        km_meds[w] = km_median(km_estimate(np.r_[dur, cw["hold_sec"].to_numpy()],
                                           np.r_[np.ones(len(dur)), np.zeros(len(cw))]))

    print("\n=== L2b  WITHIN-HOLD exit-trigger: within-position AUC, FULL inventory (0.50=no signal) ===")
    groups = onchain_groups(panel)
    all_onc = onchain_cols(panel)
    gnames = list(groups)                                       # volume sellpress holder_liq classif structural
    print(f"  {'wallet':10} {'np':>4} {'price':>6}" + "".join(f"+{g[:6]:>7}" for g in gnames)
          + f" {'+ALL':>6} {'null':>6} | onLift")
    wp_store, wp_rows = {}, []
    for w in sorted(panel["wallet"].unique()):
        wpan = panel[panel["wallet"] == w]
        pf = [f for f in NONMONO_PATH if f in wpan.columns]
        res = {"price": within_position_auc(wpan, pf)[0]}
        for g, cols in groups.items():
            res[g] = within_position_auc(wpan, pf + [c for c in cols if c in wpan.columns])[0]
        res["all"] = within_position_auc(wpan, pf + all_onc)[0]
        res["null"] = within_position_null(wpan, pf + all_onc)
        wp_store[w] = res
        print(f"  {w:10} {wpan['position_id'].nunique():>4} {res['price']:>6.3f}"
              + "".join(f"{res[g]:>8.3f}" for g in gnames)
              + f" {res['all']:>6.3f} {res['null']:>6.3f} | {res['all'] - res['price']:>+.3f}")
        wp_rows.append({"wallet": w, **{f"wp_{k}": round(v, 4) for k, v in res.items()}})
    pd.DataFrame(wp_rows).to_csv(config.OUTPUT_DIR / "c3_within_hold.csv", index=False)

    print("\n  era consistency (B-wallets, within-hold +ALL on-chain): pre vs tick")
    for w in sorted(config.B_WALLET_LABELS):
        wpan = panel[panel["wallet"] == w]
        feats = [f for f in NONMONO_PATH if f in wpan.columns] + onchain_cols(wpan)
        row = [within_position_auc(wpan[wpan["era"] == e], feats)[0]
               if wpan[wpan["era"] == e]["position_id"].nunique() >= P("exit.era_min_positions") else np.nan
               for e in ("pre", "tick")]
        print(f"    {w:10} pre={row[0]:.3f}  tick={row[1]:.3f}")

    # direction of the strongest on-chain signal at exit (position-balanced)
    print("\n  on-chain signal DIRECTION at exit (position-balanced; +higher / -lower at exit):")
    for w in sorted(panel["wallet"].unique()):
        wpan = panel[panel["wallet"] == w]
        bal = position_balanced_uni(wpan, onchain_cols(wpan))
        print(f"    {w:10} " + "; ".join(f"{f}{'+' if em > hm else '-'}{a:.2f}"
                                       for f, a, em, hm in bal[:P("exit.report_top_n")]))

    # === L3  STABILITY: temporal holdout + partial-exit sensitivity ===
    print("\n=== L3  STABILITY: temporal holdout (pooled OOS) + partial-exit sensitivity (within-hold) ===")
    pmap = partial_exit_map(ev)
    for w in sorted(panel["wallet"].unique()):
        wpan = panel[panel["wallet"] == w]
        tr, te = temporal_holdout(wpan, seg_models[(w, "all")]["full_feats"])
        pe_rate = (ev[(ev.wallet == w) & (ev.label == "PARTIAL_EXIT")].shape[0]
                   / max((ev.wallet == w).sum(), 1))
        pe = ""
        if pe_rate >= P("exit.partial_exit_min_rate"):
            pfeat = [f for f in NONMONO_PATH if f in wpan.columns] + onchain_cols(wpan)
            wp_pe = within_position_auc(add_partial_positives(wpan, pmap), pfeat)[0]
            pe = f"   partial-exit within-hold {wp_store[w]['all']:.2f}->{wp_pe:.2f} (incl. scale-outs)"
        print(f"  {w:10} temporal train/test = {tr:.2f}/{te:.2f}{pe}")

    # === VERDICT ===
    print("\n" + "=" * 84)
    print("=== C3 VERDICT (exit logic, decoded) ===")
    print("=" * 84)
    # Verdict cut-offs (tuned privately): how much a feature family must lift the
    # within-hold AUC over price alone to count as a driver, and how far above its
    # shuffle-null a signal must sit to be called decodable at all.
    vc = P("exit.verdict")
    vrows = []
    for w in sorted(panel["wallet"].unique()):
        r = wp_store[w]
        decod = r["all"] - r["null"]                       # within-hold signal above shuffle
        price_real = r["price"] - r["null"]
        drivers = [g for g in ("volume", "sellpress", "holder_liq", "classif", "structural")
                   if r.get(g, 0) - r["price"] >= vc["family_lift_min"]]
        if decod < vc["decodable_min"]:
            head = "within-hold trigger NOT decodable (duration / rotation-driven)"
        else:
            bits = (["price-path"] if price_real >= vc["decodable_min"] else []) + drivers
            head = "decodable via " + (" + ".join(bits) if bits else "combined on-chain")
        vrows.append({"wallet": w, "KM_hold_s": round(km_meds[w]),
                      "wp_price": round(r["price"], 3), "wp_onchain": round(r["all"], 3),
                      "wp_null": round(r["null"], 3), "onchain_lift": round(r["all"] - r["price"], 3),
                      "decodability": round(decod, 3), "verdict": head})
    vdf = pd.DataFrame(vrows).set_index("wallet")
    print(vdf.to_string())
    vdf.to_csv(config.OUTPUT_DIR / "c3_verdict.csv")
    print("\n  wp_price = within-hold AUC, price-shape only | wp_onchain = + on-chain dynamics")
    print("  wp_null  = within-position shuffle (must ~0.50) | decodability = wp_onchain - wp_null")
    print("  onchain_lift = wp_onchain - wp_price = the on-chain contribution to exit TIMING")
    print("\nC3 OK - survival + hazard + within-hold on-chain decoding written to analysis/outputs/.")


if __name__ == "__main__":
    _self_test()
