"""Phase 7: regime filters — can we skip the chop without losing the trends?

Filters applied on top of the baseline 22/55 signal (long only when signal AND filter):
  ADX>20 / ADX>25, MA distance (fast/slow-1 > x%), trend-strength percentile,
  vol expansion (ATR rising), and none (baseline).
Score: median Sharpe, whipsaw count (trades <8w held), exposure.
Run: python phase7_regime_filters.py -> results/phase7_report.md
"""
import os
import numpy as np
import pandas as pd
from common import load_universe, ma, atr, state_backtest, perf
from slow_turtle import RESULTS_DIR

SKIP = 110


def adx(w: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = w["High"], w["Low"], w["Close"]
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=w.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=w.index)
    trn = atr(w, n) * n
    pdi = 100 * plus.rolling(n).sum() / trn
    mdi = 100 * minus.rolling(n).sum() / trn
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.rolling(n).mean()


def whipsaws(pos: pd.Series) -> tuple[int, int]:
    """(#trades, #trades held <8 weeks) from a boolean position series."""
    d = pos.astype(int).diff()
    starts = list(np.where(d.values == 1)[0])
    ends = list(np.where(d.values == -1)[0])
    n, short = 0, 0
    for s in starts:
        e = next((x for x in ends if x > s), len(pos) - 1)
        n += 1
        short += (e - s) < 8
    return n, short


if __name__ == "__main__":
    uni = load_universe()
    results = {}
    for a, w in uni.items():
        c = w["Close"]
        fast, slow = ma(c, 22, "SMA"), ma(c, 55, "SMA")
        base = fast > slow
        dist = fast / slow - 1
        a14 = adx(w)
        strength_pct = dist.rolling(104).rank(pct=True)
        atr_ratio = atr(w, 14) / atr(w, 52)
        filters = {
            "baseline (none)": base,
            "ADX>20": base & (a14 > 20),
            "ADX>25": base & (a14 > 25),
            "MA dist>1%": base & (dist > 0.01),
            "MA dist>3%": base & (dist > 0.03),
            "strength pctile>30%": base & (strength_pct > 0.3),
            "vol expanding (ATR14>ATR52)": base & (atr_ratio > 1),
            "vol calm (ATR14<ATR52)": base & (atr_ratio < 1),
        }
        for name, sig in filters.items():
            sig = sig.fillna(False)
            r = state_backtest(w, sig)
            p = perf(r, skip=SKIP)
            n, short = whipsaws(sig.shift(2).fillna(False).iloc[SKIP:])
            results.setdefault(name, []).append(
                {"sharpe": p["sharpe"], "max_dd": p["max_dd"], "trades": n,
                 "whipsaws": short, "exposure": sig.iloc[SKIP:].mean()})
    rows = {}
    for name, lst in results.items():
        d = pd.DataFrame(lst)
        rows[name] = {"median_sharpe": d["sharpe"].median(), "median_maxdd": d["max_dd"].median(),
                      "total_trades": d["trades"].sum(), "total_whipsaws": d["whipsaws"].sum(),
                      "avg_exposure": d["exposure"].mean()}
    df = pd.DataFrame(rows).T.sort_values("median_sharpe", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = ["# Phase 7 — Regime Filters\n",
             "Filter ANDed with baseline 22/55 signal. Whipsaw = round trip held < 8 weeks.",
             "16 assets, common warmup, no costs.\n",
             df.round(2).to_markdown(), ""]
    with open(os.path.join(RESULTS_DIR, "phase7_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(df.round(2).to_string())
    print("Report -> results/phase7_report.md")
