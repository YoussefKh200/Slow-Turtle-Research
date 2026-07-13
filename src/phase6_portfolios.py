"""Phase 6: portfolio construction — measure the diversification benefit.

Portfolios (baseline signal, EWMA vol-targeted 10% per sleeve, cap 1x, from Phase 5):
  A: QQQ only | B: stock sectors | C: indices | D: global (gold+fx+oil+spy)
  E: everything (16 assets)
Outputs: correlation matrix of strategy returns, per-portfolio stats, risk contribution.
Run: python phase6_portfolios.py -> results/phase6_report.md + phase6_equity.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_universe
from stats import perf
from phase5_vol_sizing import sized_returns
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase6")
CHARTS = os.path.join(OUT, "charts")
os.makedirs(CHARTS, exist_ok=True)

SKIP = 110
PORTFOLIOS = {
    "A: QQQ only": ["QQQ"],
    "B: Stock sectors": ["TECH", "UTILITIES", "INDUSTRIALS", "MATERIALS"],
    "C: Indices": ["SPY", "QQQ", "DIA", "IWM", "NAS100", "SP500"],
    "D: Global macro": ["SPY", "GOLD", "SILVER", "OIL", "EURUSD", "GBPUSD", "USDJPY"],
    "E: Everything": None,  # all 16
}
# fixed categorical colors, one per portfolio (identity, never cycled)
COLORS = {"A: QQQ only": "#4269d0", "B: Stock sectors": "#efb118", "C: Indices": "#ff725c",
          "D: Global macro": "#6cc5b0", "E: Everything": "#9c6b4e"}


def strategy_returns() -> pd.DataFrame:
    uni = load_universe()
    rets = {a: sized_returns(w, "ewma", 0.10, 1).iloc[SKIP:] for a, w in uni.items()}
    return pd.concat(rets, axis=1, sort=True).dropna()  # common window


def risk_contribution(r: pd.DataFrame) -> pd.Series:
    w = np.ones(r.shape[1]) / r.shape[1]
    cov = r.cov().values
    mrc = cov @ w
    rc = w * mrc / (w @ cov @ w)
    return pd.Series(rc, index=r.columns)


if __name__ == "__main__":
    R = strategy_returns()
    rows, curves = {}, {}
    for name, assets in PORTFOLIOS.items():
        sub = R if assets is None else R[assets]
        port = sub.mean(axis=1)
        p = perf(port)
        rows[name] = {**p, "ann_vol": port.std() * np.sqrt(52),
                      "avg_pair_corr": sub.corr().values[np.triu_indices(sub.shape[1], 1)].mean()
                      if sub.shape[1] > 1 else 1.0}
        curves[name] = (1 + port).cumprod()
    stats = pd.DataFrame(rows).T

    # equity chart: log scale, direct labels, thin lines, recessive grid
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, lw=2, color=COLORS[name], label=name)
        ax.annotate(name.split(":")[0], (eq.index[-1], eq.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", color=COLORS[name], fontsize=9, va="center")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Portfolio equity (vol-targeted sleeves, equal weight, log scale)", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(os.path.join(CHARTS, "equity.png"), dpi=150)
    plt.close(fig)

    corr = R.corr()
    rc = risk_contribution(R)
    lines = [
        "# Phase 6 — Portfolio Construction\n",
        f"Common window {R.index[0].date()} to {R.index[-1].date()}. Sleeves = baseline signal,",
        "EWMA vol target 10%, cap 1x. Equal-weight across sleeves, cash earns 0.\n",
        "## Portfolio comparison\n", stats.round(2).to_markdown(), "",
        "![equity](charts/equity.png)\n",
        "## Risk contribution, Portfolio E (equal weight)\n",
        rc.sort_values(ascending=False).round(3).to_frame("risk_share").to_markdown(), "",
        "## Strategy-return correlation matrix\n", corr.round(2).to_markdown(), "",
    ]
    with open(os.path.join(OUT, "PHASE6_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(stats.round(2).to_string())
    print("\nAvg pairwise corr, all 16 sleeves:", round(corr.values[np.triu_indices(16, 1)].mean(), 2))
    print("Report -> results/phase6/PHASE6_REPORT.md")
