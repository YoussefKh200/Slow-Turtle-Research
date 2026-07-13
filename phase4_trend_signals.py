"""Phase 4: which trend-identification method finds persistent trends best?

Families: Donchian breakouts, time-series momentum, MA cross (Phase 1 baseline),
trend-strength measures (efficiency ratio, regression R2/slope, Hurst) used as signals.
All long-only, weekly, same execution/lag as baseline. Score = median Sharpe across assets.
Run: python phase4_trend_signals.py -> results/phase4_report.md
"""
import os
import numpy as np
import pandas as pd
from common import load_universe, ma, state_backtest, perf
from slow_turtle import RESULTS_DIR

SKIP = 110  # common warmup (longest lookback 104w + lag)


def donchian(close: pd.Series, n: int) -> pd.Series:
    """Long when price at n-week high; flat when it hits n//2-week low (classic turtle exit)."""
    hi = close.rolling(n).max()
    lo = close.rolling(max(2, n // 2)).min()
    sig = pd.Series(np.nan, index=close.index)
    sig[close >= hi] = 1.0
    sig[close <= lo] = 0.0
    return sig.ffill().fillna(0.0).astype(bool)


def tsmom(close: pd.Series, weeks: int, skip_last: int = 0) -> pd.Series:
    return close.shift(skip_last) / close.shift(weeks) - 1 > 0


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    return (close - close.shift(n)).abs() / close.diff().abs().rolling(n).sum()


def reg_stats(close: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """Rolling regression of log price on time: (slope per week, R2)."""
    y = np.log(close)
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    sxx = (x ** 2).sum()

    def _slope(v):
        return np.dot(x, v - v.mean()) / sxx

    def _r2(v):
        b = np.dot(x, v - v.mean()) / sxx
        res = v - v.mean() - b * x
        tot = ((v - v.mean()) ** 2).sum()
        return 1 - res @ res / tot if tot > 0 else 0.0

    return y.rolling(n).apply(_slope, raw=True), y.rolling(n).apply(_r2, raw=True)


def hurst(close: pd.Series, n: int = 104) -> pd.Series:
    """Rolling Hurst via variance ratio of log returns (lag 1 vs lag 4)."""
    r = np.log(close).diff()

    def _h(v):
        v = v[~np.isnan(v)]
        if len(v) < 20:
            return np.nan
        v4 = v[len(v) % 4:].reshape(-1, 4).sum(axis=1)
        var1, var4 = v.var(), v4.var()
        if var1 <= 0 or var4 <= 0:
            return np.nan
        return 0.5 * np.log2(var4 / var1) / 2  # H = log(var ratio)/(2*log(lag))

    return r.rolling(n).apply(_h, raw=True)


def signals_for(w: pd.DataFrame) -> dict[str, pd.Series]:
    c = w["Close"]
    slope52, r2_52 = reg_stats(c, 52)
    er26 = efficiency_ratio(c, 26)
    out = {
        "MA 22/55 (baseline)": ma(c, 22, "SMA") > ma(c, 55, "SMA"),
        "Donchian 20w": donchian(c, 20),
        "Donchian 40w (~200d)": donchian(c, 40),
        "Donchian 50w": donchian(c, 50),
        "Donchian 100w": donchian(c, 100),
        "Mom 6m": tsmom(c, 26),
        "Mom 9m": tsmom(c, 39),
        "Mom 12m": tsmom(c, 52),
        "Mom 12-1": tsmom(c, 52, skip_last=4),
        "Slope52>0 & R2>0.5": (slope52 > 0) & (r2_52 > 0.5),
        "Slope52>0": slope52 > 0,
        "ER26>0.3 & up": (er26 > 0.3) & (c > c.shift(26)),
        "Hurst>0.5 & up": (hurst(c) > 0.5) & (c > c.shift(26)),
    }
    return {k: v.fillna(False) for k, v in out.items()}


if __name__ == "__main__":
    uni = load_universe()
    rows = {}
    for a, w in uni.items():
        for name, sig in signals_for(w).items():
            p = perf(state_backtest(w, sig), skip=SKIP)
            rows.setdefault(name, {})[a] = p["sharpe"]
    df = pd.DataFrame(rows).T  # signals x assets
    summary = pd.DataFrame({
        "median_sharpe": df.median(axis=1),
        "mean_sharpe": df.mean(axis=1),
        "worst_asset": df.min(axis=1),
        "pct_assets_positive": (df > 0).mean(axis=1),
    }).sort_values("median_sharpe", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.round(2).to_csv(os.path.join(RESULTS_DIR, "phase4_sharpe_by_asset.csv"))
    lines = [
        "# Phase 4 — Trend Identification Methods\n",
        f"Long-only, weekly, next-Monday-open execution, common warmup {SKIP} weeks, no costs.",
        "Score = median Sharpe across 16 assets.\n",
        "## Summary (sorted)\n", summary.round(2).to_markdown(), "",
        "## Sharpe by asset\n", df.round(2).to_markdown(), "",
    ]
    with open(os.path.join(RESULTS_DIR, "phase4_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(summary.round(2).to_string())
    print("Report -> results/phase4_report.md")
