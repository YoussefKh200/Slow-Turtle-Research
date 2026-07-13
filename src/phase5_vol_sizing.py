"""Phase 5: volatility-adjusted position sizing vs the book's full-position sizing.

Sizing methods: fixed 1x (baseline), ATR-based, realized vol, EWMA vol — each scaled
to hit an annual vol target (10/15/20%) while in a trend. Weight capped at 1 (no leverage)
and at 2x (mild leverage) to show both. Score per asset + equal-weight portfolio of all 16.
Run: python phase5_vol_sizing.py -> results/phase5_report.md
"""
import os
import numpy as np
import pandas as pd
from data import load_universe
from backtest import ma, atr, WEEKS_PER_YEAR
from stats import perf
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase5")

SKIP = 110


def weekly_vol(close: pd.Series, kind: str) -> pd.Series:
    r = close.pct_change()
    if kind == "realized26":
        return r.rolling(26).std() * np.sqrt(WEEKS_PER_YEAR)
    if kind == "ewma":
        return r.ewm(span=26, adjust=False).std() * np.sqrt(WEEKS_PER_YEAR)
    raise ValueError(kind)


def sized_returns(w: pd.DataFrame, sizing: str, target: float, cap: float) -> pd.Series:
    c = w["Close"]
    sig = (ma(c, 22, "SMA") > ma(c, 55, "SMA"))
    if sizing == "fixed":
        weight = pd.Series(1.0, index=c.index)
    elif sizing == "atr":
        av = atr(w, 14) / c * np.sqrt(WEEKS_PER_YEAR)  # ATR as % of price, annualized
        weight = (target / av).clip(upper=cap)
    else:
        weight = (target / weekly_vol(c, sizing)).clip(upper=cap)
    ret_oo = w["Open"].pct_change()
    pos = (sig * weight).shift(2).fillna(0.0)
    return ret_oo.fillna(0.0) * pos


if __name__ == "__main__":
    uni = load_universe()
    configs = [("fixed 1x", "fixed", 0, 1)] + [
        (f"{s} t={t:.0%} cap={c}x", s, t, c)
        for s in ("atr", "realized26", "ewma")
        for t in (0.10, 0.15, 0.20)
        for c in (1, 2)
    ]
    per_asset, portfolio = {}, {}
    for label, sizing, target, cap in configs:
        rets = {a: sized_returns(w, sizing, target, cap).iloc[SKIP:] for a, w in uni.items()}
        sharpes = {a: perf(r)["sharpe"] for a, r in rets.items()}
        per_asset[label] = pd.Series(sharpes)
        # equal-weight portfolio, common window only (all assets past their warmup)
        port = pd.concat(rets, axis=1, sort=True).dropna().mean(axis=1)
        p = perf(port)
        portfolio[label] = {"port_sharpe": p["sharpe"], "port_cagr": p["cagr"],
                            "port_maxdd": p["max_dd"], "port_calmar": p["calmar"],
                            "median_asset_sharpe": per_asset[label].median()}
    dfp = pd.DataFrame(portfolio).T.sort_values("port_sharpe", ascending=False)

    os.makedirs(OUT, exist_ok=True)
    lines = [
        "# Phase 5 — Volatility-Adjusted Sizing\n",
        "Baseline signal (SMA 22/55) held constant; only position size changes.",
        "Portfolio = equal-weight of all 16 assets' strategy returns (cash earns 0).\n",
        "## Portfolio impact (sorted by Sharpe)\n", dfp.round(2).to_markdown(), "",
        "## Per-asset Sharpe by sizing\n",
        pd.DataFrame(per_asset).round(2).to_markdown(), "",
    ]
    with open(os.path.join(OUT, "PHASE5_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(dfp.round(2).to_string())
    print("Report -> results/phase5/PHASE5_REPORT.md")
