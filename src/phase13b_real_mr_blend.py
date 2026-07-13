"""Phase 13b: re-run the Phase 13 TF+MR blend with the *real* MR sleeve.

Phase 13 (see phase13_combine_mr.py) rejected blending because its MR sleeve
was a deliberately simple placeholder (Sharpe 0.22) built only to measure
correlation. RESEARCH.md's conclusion: revisit once the dedicated
Mean-Reversion-Research program produces a sleeve with Sharpe 0.7+.

That program is done: ../../MEAN REVERSION (sibling repo) reports MR_QQQ /
MR_SPY at Sharpe 0.813 solo (Z-score<-2 entry, 6% MAE stop, 10% vol target).
This script imports that program's own sleeve functions directly (no
reimplementation, so the two repos can't drift apart), resamples its daily
returns to the weekly W-FRI calendar TF runs on, and re-does the blend grid.

Run: python phase13b_real_mr_blend.py -> results/phase13b/PHASE13B_REPORT.md
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MR_REPO_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "MEAN REVERSION", "src")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase13b")
CHARTS = os.path.join(OUT, "charts")
os.makedirs(CHARTS, exist_ok=True)


def daily_to_weekly(r: pd.Series) -> pd.Series:
    """Compound daily returns into W-FRI weekly returns, matching TF's calendar."""
    return (1 + r).resample("W-FRI").apply(lambda x: x.prod() - 1)


def load_real_mr_weekly() -> pd.Series:
    """Import the MR program's own sleeve code (unmodified) from the sibling repo.

    Both repos have same-named data.py/backtest.py modules, so this repo's
    versions must not be imported yet when MR's are loaded -- Python caches
    modules by name in sys.modules, and whichever loads first wins. Import
    MR's modules first, pull what's needed, then evict them from the cache
    so this script's own `from data import ...` etc. below get TF's modules.
    """
    if not os.path.isdir(MR_REPO_SRC):
        raise SystemExit(f"Mean-Reversion-Research repo not found at {MR_REPO_SRC} "
                          "-- this phase needs it as a sibling directory.")
    sys.path.insert(0, MR_REPO_SRC)
    import phase13_portfolio as mr  # noqa: the MR program's own sleeve code

    mr_daily = {"MR_QQQ": mr.vol_target(mr.mr_returns("QQQ")),
                "MR_SPY": mr.vol_target(mr.mr_returns("SPY"))}
    weekly = pd.concat({k: daily_to_weekly(v) for k, v in mr_daily.items()},
                        axis=1, sort=True).dropna().mean(axis=1)

    for modname in ("data", "backtest", "phase13_portfolio"):
        sys.modules.pop(modname, None)
    sys.path.remove(MR_REPO_SRC)
    return weekly


if __name__ == "__main__":
    mr_weekly = load_real_mr_weekly()

    from stats import perf, sharpe
    from phase12_robustness import portfolio, ftmo

    tf = ftmo(portfolio())  # FTMO-configured weekly trend portfolio (same as phase13)

    common = tf.index.intersection(mr_weekly.index)
    tf, mr_w = tf.loc[common], mr_weekly.loc[common]
    mr_w = mr_w * (tf.std() / mr_w.std())  # vol-match to TF for a fair blend

    corr = tf.corr(mr_w)
    rows, curves = {}, {}
    for wtf in [1.0, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0]:
        blend = wtf * tf + (1 - wtf) * mr_w
        p = perf(blend)
        rows[f"TF {wtf:.0%} / MR {1-wtf:.0%}"] = {
            "sharpe": sharpe(blend), "cagr": p["cagr"], "max_dd": p["max_dd"], "calmar": p["calmar"]}
        if wtf in (1.0, 0.7, 0.5, 0.0):
            curves[f"TF {wtf:.0%} / MR {1-wtf:.0%}"] = (1 + blend).cumprod()
    df = pd.DataFrame(rows).T

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for (name, eq), col in zip(curves.items(), ["#4269d0", "#efb118", "#ff725c", "#6cc5b0"]):
        ax.plot(eq.index, eq.values, lw=2, color=col, label=name)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"TF + real MR blends (strategy correlation = {corr:.2f})", fontsize=11)
    fig.savefig(os.path.join(CHARTS, "blend.png"), dpi=150)
    plt.close(fig)

    lines = ["# Phase 13b — Trend Following + real Mean Reversion sleeve\n",
             "TF = FTMO-configured trend portfolio (same as phase13). MR = MR_QQQ + MR_SPY",
             "from the Mean-Reversion-Research program (Sharpe 0.813 solo, Phase 9 sizing),",
             "daily returns compounded to weekly, vol-matched to TF.",
             f"**Correlation TF vs real MR: {corr:.2f}**\n",
             df.round(3).to_markdown(), "",
             "![blend](charts/blend.png)\n"]
    with open(os.path.join(OUT, "PHASE13B_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"TF/real-MR correlation: {corr:.2f}")
    print(df.round(3).to_string())
    print("Report -> results/phase13b/PHASE13B_REPORT.md")
