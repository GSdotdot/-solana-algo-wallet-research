"""C6 — FULL indicator parameter sweep: which TA setting clusters at entries/exits?

Comprehensive, not a hand-picked subset (an earlier pass swept a small grid). Builds a
per-token 1m OHLCV series = the pre-discovery price history (o/h/l/c/v) + the snapshot
price resampled to 1m (o=first, h=max, l=min, c=last, v=1m volume), then computes the
STANDARD TA LIBRARY across all five categories over a ladder of periods, reads each
causally at entries/exits, and contrasts vs a universe baseline (random bars of the
same tokens). Per (indicator, period): single-feature AUC with a TOKEN-HOLDOUT test
(train-selected -> test-evaluated) so the tightest cluster isn't a grid-search artifact.

Indicators (period-swept unless noted; the period ladder is in config, redacted):
  momentum   : rsi, stoch_k, stoch_d, stochrsi, williams_r, cci, roc, momentum, cmo, trix, rvi,
               fisher; macd_hist, tsi, ppo, awesome_osc, ultimate_osc (textbook fixed params)
  trend      : sma_dist, ema_dist, wma_dist, hma_dist, adx, plus_di, minus_di, aroon_up,
               aroon_down, aroon_osc, vortex_pos, vortex_neg, vortex_diff, dpo, linreg_slope, vhf
  volatility : bb_pctb, bb_width, keltner_pos, frhigh, frlow, atr_norm, std_norm, ulcer
  volume     : obv_slope, cmf, mfi, force_index, eom, vol_roc, vol_zscore
  statistical: zscore, price_pctile, up_ratio, ret_skew, ret_kurt
  NOT covered (too fiddly to hand-trust): Parabolic SAR, Supertrend, KAMA, Ichimoku, Klinger.

The constants left inline in build_panel (smoothing lengths inside an indicator's
definition, band widths, the fixed-parameter classics) are the textbook definitions
of those indicators, not tuned values.

Run `python -m analysis.indicators`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config, data, features

warnings.filterwarnings("ignore")

P = config.P


def _periods() -> tuple:
    """The swept period ladder (bars); tuned privately."""
    return tuple(P("indicators.periods"))


# --- small helpers (vectorised) --------------------------------------------------
def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def _rma(s, n):  return s.ewm(alpha=1 / n, adjust=False).mean()        # Wilder
def _sma(s, n):  return s.rolling(n, min_periods=2).mean()
def _hhv(s, n):  return s.rolling(n, min_periods=2).max()
def _llv(s, n):  return s.rolling(n, min_periods=2).min()


def _wma(s, n):
    a = s.to_numpy(float); out = np.full(len(a), np.nan)
    if len(a) >= n:
        w = np.arange(1, n + 1, dtype=float)
        out[n - 1:] = np.convolve(a, w, "valid") / w.sum()
    return pd.Series(out, index=s.index)


def _linreg_slope(c, n):
    """Rolling least-squares slope (t=0..n-1) normalised by the window mean. Vectorised."""
    a = c.to_numpy(float); out = np.full(len(a), np.nan)
    if len(a) >= n:
        sty = np.convolve(a, np.arange(n, dtype=float), "valid")       # Σ t*y
        sy = np.convolve(a, np.ones(n), "valid")                       # Σ y
        slope = (sty - (n - 1) / 2 * sy) / (n * (n * n - 1) / 12)
        out[n - 1:] = slope / (sy / n + 1e-18)
    return pd.Series(out, index=c.index)


def build_panel(o, h, l, c, v) -> pd.DataFrame:
    """The full TA library as a wide DataFrame indexed like the inputs."""
    out = {}
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    tp = (h + l + c) / 3
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    ret = c.pct_change()

    for n in _periods():
        hh, ll = _hhv(h, n), _llv(l, n)
        rng = (hh - ll).replace(0, np.nan)
        atr = _rma(tr, n)
        # --- momentum ---
        out[f"rsi_{n}"] = 100 - 100 / (1 + _rma(up, n) / (_rma(dn, n) + 1e-12))
        out[f"stoch_k_{n}"] = (c - ll) / rng * 100
        out[f"stoch_d_{n}"] = _sma(out[f"stoch_k_{n}"], 3)
        rsi = out[f"rsi_{n}"]
        out[f"stochrsi_{n}"] = (rsi - _llv(rsi, n)) / (_hhv(rsi, n) - _llv(rsi, n) + 1e-12)
        out[f"williams_r_{n}"] = (hh - c) / rng * -100
        out[f"cci_{n}"] = (tp - _sma(tp, n)) / (0.015 * (tp - _sma(tp, n)).abs().rolling(n, min_periods=2).mean() + 1e-12)
        out[f"roc_{n}"] = c / c.shift(n) - 1
        out[f"momentum_{n}"] = (c - c.shift(n)) / c
        out[f"cmo_{n}"] = 100 * (up.rolling(n).sum() - dn.rolling(n).sum()) / (up.rolling(n).sum() + dn.rolling(n).sum() + 1e-12)
        tx = _ema(_ema(_ema(np.log(c.clip(lower=1e-18)), n), n), n)
        out[f"trix_{n}"] = tx.diff()
        num = _ema(_ema(c - o, n), 4); den = _ema(_ema(h - l, n), 4)
        out[f"rvi_{n}"] = num / (den + 1e-12)
        x = (2 * ((c - ll) / rng - 0.5)).clip(-0.999, 0.999)
        xs = _ema(x, 5)
        out[f"fisher_{n}"] = 0.5 * np.log((1 + xs) / (1 - xs))
        # --- trend ---
        out[f"sma_dist_{n}"] = c / _sma(c, n) - 1
        out[f"ema_dist_{n}"] = c / _ema(c, n) - 1
        out[f"wma_dist_{n}"] = c / _wma(c, n) - 1
        hma = _wma(2 * _wma(c, max(2, n // 2)) - _wma(c, n), max(2, int(np.sqrt(n))))
        out[f"hma_dist_{n}"] = c / hma - 1
        up_move, dn_move = h.diff(), -l.diff()
        plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
        pdi = 100 * _rma(pd.Series(plus_dm, index=c.index), n) / (atr + 1e-12)
        mdi = 100 * _rma(pd.Series(minus_dm, index=c.index), n) / (atr + 1e-12)
        out[f"plus_di_{n}"], out[f"minus_di_{n}"] = pdi, mdi
        out[f"adx_{n}"] = _rma(100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12), n)
        out[f"aroon_up_{n}"] = h.rolling(n + 1, min_periods=2).apply(lambda a: a.argmax() / (len(a) - 1) * 100, raw=True)
        out[f"aroon_down_{n}"] = l.rolling(n + 1, min_periods=2).apply(lambda a: a.argmin() / (len(a) - 1) * 100, raw=True)
        out[f"aroon_osc_{n}"] = out[f"aroon_up_{n}"] - out[f"aroon_down_{n}"]
        vp = (h - l.shift()).abs(); vm = (l - h.shift()).abs()
        out[f"vortex_pos_{n}"] = vp.rolling(n).sum() / (tr.rolling(n).sum() + 1e-12)
        out[f"vortex_neg_{n}"] = vm.rolling(n).sum() / (tr.rolling(n).sum() + 1e-12)
        out[f"vortex_diff_{n}"] = out[f"vortex_pos_{n}"] - out[f"vortex_neg_{n}"]
        out[f"dpo_{n}"] = c.shift(n // 2 + 1) / _sma(c, n) - 1
        out[f"linreg_slope_{n}"] = _linreg_slope(c, n)
        out[f"vhf_{n}"] = (hh - ll).abs() / (c.diff().abs().rolling(n).sum() + 1e-12)
        # --- volatility / bands ---
        ma, sd = _sma(c, n), c.rolling(n, min_periods=2).std()
        out[f"bb_pctb_{n}"] = (c - (ma - 2 * sd)) / (4 * sd + 1e-12)
        out[f"bb_width_{n}"] = 4 * sd / (ma + 1e-12)
        out[f"keltner_pos_{n}"] = (c - _ema(c, n)) / (atr + 1e-12)
        out[f"frhigh_{n}"] = c / hh - 1
        out[f"frlow_{n}"] = c / ll - 1
        out[f"atr_norm_{n}"] = atr / c
        out[f"std_norm_{n}"] = ret.rolling(n, min_periods=2).std()
        dd = (c / _hhv(c, n) - 1) * 100
        out[f"ulcer_{n}"] = np.sqrt((dd ** 2).rolling(n, min_periods=2).mean())
        # --- volume ---
        obv = (np.sign(d).fillna(0) * v).cumsum()
        out[f"obv_slope_{n}"] = (obv - obv.shift(n)) / (v.rolling(n).sum() + 1e-9)
        mfm = ((c - l) - (h - c)) / rng
        out[f"cmf_{n}"] = (mfm * v).rolling(n).sum() / (v.rolling(n).sum() + 1e-9)
        rmf = tp * v
        pos_mf = rmf.where(tp > tp.shift(), 0).rolling(n).sum()
        neg_mf = rmf.where(tp < tp.shift(), 0).rolling(n).sum()
        out[f"mfi_{n}"] = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-9))
        out[f"force_index_{n}"] = _ema(d * v, n) / (c * v.rolling(n).mean() + 1e-9)
        out[f"eom_{n}"] = (((h + l) / 2).diff() * (h - l) / (v + 1e-9)).rolling(n).mean()
        out[f"vol_roc_{n}"] = v / (v.shift(n) + 1e-9) - 1
        out[f"vol_zscore_{n}"] = (v - v.rolling(n, min_periods=2).mean()) / (v.rolling(n, min_periods=2).std() + 1e-9)
        # --- statistical / position ---
        out[f"zscore_{n}"] = (c - ma) / (sd + 1e-12)
        out[f"price_pctile_{n}"] = c.rolling(n, min_periods=2).rank(pct=True)
        out[f"up_ratio_{n}"] = (ret > 0).rolling(n, min_periods=2).mean()
        out[f"ret_skew_{n}"] = ret.rolling(n, min_periods=4).skew()
        out[f"ret_kurt_{n}"] = ret.rolling(n, min_periods=4).kurt()

    # --- fixed-parameter classics ---
    out["macd_hist"] = (_ema(c, 12) - _ema(c, 26) - _ema(_ema(c, 12) - _ema(c, 26), 9)) / c
    pc = c.diff()
    out["tsi"] = 100 * _ema(_ema(pc, 25), 13) / (_ema(_ema(pc.abs(), 25), 13) + 1e-12)
    out["ppo"] = (_ema(c, 12) - _ema(c, 26)) / (_ema(c, 26) + 1e-12) * 100
    med = (h + l) / 2
    out["awesome_osc"] = _sma(med, 5) - _sma(med, 34)
    bp = c - pd.concat([l, c.shift()], axis=1).min(axis=1)
    for k in (7, 14, 28):
        out[f"_uo_bp{k}"] = bp.rolling(k).sum() / (tr.rolling(k).sum() + 1e-12)
    out["ultimate_osc"] = 100 * (4 * out.pop("_uo_bp7") + 2 * out.pop("_uo_bp14") + out.pop("_uo_bp28")) / 7
    return pd.DataFrame(out, index=c.index)


def token_ohlcv_1m(tok, bf_json, snaps):
    parts = []
    bf = features.parse_backfill(bf_json) if bf_json else pd.DataFrame()
    if len(bf) and {"o", "h", "l", "c"} <= set(bf.columns):
        b = bf[["o", "h", "l", "c", "v"]].apply(pd.to_numeric, errors="coerce")
        b["t"] = pd.to_numeric(bf["t"], errors="coerce") // 60 * 60
        parts.append(b)
    sf = snaps.get(tok)
    if sf is not None and len(sf):
        p = pd.to_numeric(sf["price"], errors="coerce")
        t = (sf["timestamp"] // 60 * 60).to_numpy()
        g = pd.DataFrame({"t": t, "p": p.to_numpy(),
                          "vol": pd.to_numeric(sf.get("volume_1m_usd", 0), errors="coerce").to_numpy()})
        agg = g.groupby("t").agg(o=("p", "first"), h=("p", "max"), l=("p", "min"),
                                 c=("p", "last"), v=("vol", "last")).reset_index()
        parts.append(agg)
    if not parts:
        return None
    df = (pd.concat(parts).dropna(subset=["c"]).query("c > 0")
          .groupby("t").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                            c=("c", "last"), v=("v", "sum")).sort_index())
    return df if len(df) >= P("indicators.min_bars") else None


def values_at(panels, tt):
    cols, out = None, []
    for tok, g in tt.groupby("token_address", sort=False):
        p = panels.get(tok)
        if p is None:
            out.append(pd.DataFrame(index=g.index)); continue
        cols = p.columns
        pos = np.searchsorted(p.index.values, g["t"].to_numpy(), side="right") - 1
        v = p.iloc[np.clip(pos, 0, len(p) - 1)].reset_index(drop=True)
        v.index = g.index
        v[pos < 0] = np.nan
        out.append(v)
    res = pd.concat(out).reindex(tt.index)
    return res.reindex(columns=cols) if cols is not None else res


def _auc(e, b):
    x = pd.concat([e, b]); y = np.r_[np.ones(len(e)), np.zeros(len(b))]
    m = x.notna().to_numpy() & np.isfinite(x.to_numpy())
    lim = P("indicators.auc_min")           # {total, per_class}; tuned privately
    if m.sum() < lim["total"] or y[m].sum() < lim["per_class"] or (1 - y[m]).sum() < lim["per_class"]:
        return np.nan
    try:
        return roc_auc_score(y[m], x[m].to_numpy())
    except ValueError:
        return np.nan


def sweep(ev_vals, base, ind_cols):
    """Per wallet x indicator-col: AUC(entry vs baseline), token-holdout train/test."""
    rng = np.random.default_rng(0)
    rows = []
    for w, g in ev_vals.groupby("wallet"):
        toks = g["token_address"].unique()
        tr = set(rng.choice(toks, size=max(1, int(len(toks) * P("indicators.train_token_frac"))), replace=False))
        gtr, gte = g[g["token_address"].isin(tr)], g[~g["token_address"].isin(tr)]
        btr = base[base["token_address"].isin(tr)]; bte = base[~base["token_address"].isin(tr)]
        for col in ind_cols:
            a_tr = _auc(gtr[col], btr[col])
            if a_tr != a_tr:
                continue
            rows.append({"wallet": w, "indicator": col, "auc_train": a_tr,
                         "auc_test": _auc(gte[col], bte[col]),
                         "ev_median": g[col].median(), "base_median": base[col].median()})
    return pd.DataFrame(rows)


def _self_test() -> None:
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    ev = data.load_events()
    con = data._conn()
    bfm = dict(pd.read_sql("SELECT token_address, ohlcv_backfill_json FROM token_identity", con).values)
    con.close()

    print("building per-token 1m OHLCV + FULL indicator panels ...")
    snaps = features.load_snapshot_frames()
    toks = set(ev["token_address"]) | set(bfm)
    panels = {}
    for t in toks:
        oh = token_ohlcv_1m(t, bfm.get(t), snaps)
        if oh is not None:
            panels[t] = build_panel(oh["o"], oh["h"], oh["l"], oh["c"], oh["v"])
    ncols = len(next(iter(panels.values())).columns)
    print(f"  tokens with a usable panel: {len(panels):,}  | indicators-x-periods per token: {ncols}")

    ent = ev[ev["label"].isin(("ENTRY", "FLIP"))][["wallet", "token_address", "timestamp"]].copy()
    ent["t"] = ent["timestamp"]
    xit = ev[ev["label"] == "EXIT"][["wallet", "token_address", "timestamp"]].copy()
    xit["t"] = xit["timestamp"]
    rng = np.random.default_rng(0)
    bp = [t for t in panels if t in set(ev["token_address"])]
    pick = rng.choice(len(bp), size=P("indicators.baseline_samples"))
    brows = []
    for ti in np.unique(pick):
        p = panels[bp[ti]]
        r = p.iloc[rng.integers(0, len(p), size=int((pick == ti).sum()))].copy()
        r["token_address"] = bp[ti]; brows.append(r)
    base = pd.concat(brows, ignore_index=True)
    ind_cols = [c for c in next(iter(panels.values())).columns]

    ent_v = pd.concat([ent[["wallet", "token_address"]].reset_index(drop=True),
                       values_at(panels, ent).reset_index(drop=True)], axis=1)
    xit_v = pd.concat([xit[["wallet", "token_address"]].reset_index(drop=True),
                       values_at(panels, xit).reset_index(drop=True)], axis=1)
    print(f"  entry coverage {ent_v[ind_cols[0]].notna().mean():.0%} | baseline n={len(base):,} | "
          f"sweeping {len(ind_cols)} indicator-params x {ent_v['wallet'].nunique()} wallets x 2 phases\n")

    for vals, lab in [(ent_v, "ENTRY"), (xit_v, "EXIT")]:
        sw = sweep(vals, base, ind_cols)
        sw["abs"] = (sw["auc_train"] - 0.5).abs()
        # require OOS validation: test AUC agrees in direction, and keeps at least a
        # set share of the train-set effect size (the share is tuned privately)
        v = P("indicators.validation")      # {min_test_share, strong_abs_auc}
        sw["validated"] = (np.sign(sw["auc_train"] - 0.5) == np.sign(sw["auc_test"] - 0.5)) & \
                          ((sw["auc_test"] - 0.5).abs() >= v["min_test_share"] * sw["abs"])
        sw.round(4).to_csv(config.OUTPUT_DIR / f"c6_{lab.lower()}_sweep_full.csv", index=False)
        print(f"=== C6 {lab}: TOP VALIDATED indicators per wallet (of {len(ind_cols)} swept) ===")
        for w in sorted(sw["wallet"].unique()):
            g = sw[(sw.wallet == w) & sw.validated].sort_values("abs", ascending=False)
            tops = "; ".join(f"{r['indicator']}={r['auc_train']:.2f}/{r['auc_test']:.2f}"
                             f"({'+' if r['ev_median'] > r['base_median'] else '-'})"
                             for _, r in g.head(P("indicators.report_top_n")).iterrows())
            print(f"  {w:10} {tops if tops else '(none cleared OOS)'}")
        print(f"\n  {lab}: indicator FAMILIES that NEVER cleared the strong-effect bar OOS for any wallet:")
        strong = sw[sw.validated & (sw["abs"] >= v["strong_abs_auc"])]
        fams = sorted({c.rsplit("_", 1)[0] for c in ind_cols})
        worked = {r["indicator"].rsplit("_", 1)[0] for _, r in strong.iterrows()}
        print("    " + ", ".join(f for f in fams if f not in worked))
        print()

    print("C6 OK - FULL indicator sweep written (c6_entry_sweep_full / c6_exit_sweep_full.csv).")


if __name__ == "__main__":
    _self_test()
