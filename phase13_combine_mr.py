"""Phase 13: combine trend following with a simple mean-reversion sleeve.

MR sleeve (weekly, long-only, equity indices): buy when close < 26w SMA AND 2-week
return < -5% (panic dip inside a non-crashed market: close > 55w SMA * 0.85);
exit when close > 26w SMA or after 4 weeks. Deliberately simple — the point is the
correlation, not MR alpha maximization.
Allocator: vol-weighted blends TF:MR from 100:0 to 0:100.
Run: python phase13_combine_mr.py -> results/phase13_report.md + phase13_blend.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_universe, ma, perf, WEEKS_PER_YEAR
from phase12_robustness import portfolio, ftmo, sharpe
from slow_turtle import RESULTS_DIR

SKIP = 110
MR_ASSETS = ["SPY", "QQQ", "DIA", "IWM", "NAS100", "SP500", "TECH", "INDUSTRIALS"]


def mr_returns(w: pd.DataFrame) -> pd.Series:
    c, o = w["Close"], w["Open"]
    sma26, sma55 = ma(c, 26, "SMA"), ma(c, 55, "SMA")
    dip = (c < sma26) & (c.pct_change(2) < -0.05) & (c > sma55 * 0.85)
    pos = np.zeros(len(w))
    held = 0
    in_pos = False
    for i in range(len(w)):
        if in_pos:
            held += 1
            if c.iloc[i] > sma26.iloc[i] or held >= 4:
                in_pos = False
        if not in_pos and bool(dip.iloc[i]):
            in_pos, held = True, 0
        pos[i] = in_pos
    ret_oo = o.pct_change().fillna(0.0)
    return (ret_oo * pd.Series(pos, index=w.index).shift(2)).fillna(0.0)


if __name__ == "__main__":
    uni = load_universe()
    tf = ftmo(portfolio())  # FTMO-configured trend portfolio
    mr_sleeves = {a: mr_returns(uni[a]).iloc[SKIP:] for a in MR_ASSETS}
    mr = pd.concat(mr_sleeves, axis=1, sort=True).dropna().mean(axis=1)
    # scale MR to similar vol as TF for a fair blend
    common = tf.index.intersection(mr.index)
    tf, mr = tf.loc[common], mr.loc[common]
    mr = mr * (tf.std() / mr.std())

    corr = tf.corr(mr)
    rows, curves = {}, {}
    for wtf in [1.0, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0]:
        blend = wtf * tf + (1 - wtf) * mr
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
    ax.set_title(f"TF + MR blends (strategy correlation = {corr:.2f})", fontsize=11)
    fig.savefig(os.path.join(RESULTS_DIR, "phase13_blend.png"), dpi=150)
    plt.close(fig)

    lines = ["# Phase 13 — Trend Following + Mean Reversion\n",
             f"TF = FTMO-configured trend portfolio. MR = weekly panic-dip sleeve on {len(MR_ASSETS)}",
             f"equity indices, vol-matched to TF. **Correlation TF vs MR: {corr:.2f}**\n",
             df.round(2).to_markdown(), "",
             "![blend](phase13_blend.png)\n"]
    with open(os.path.join(RESULTS_DIR, "phase13_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"TF/MR correlation: {corr:.2f}")
    print(df.round(2).to_string())
    print("Report -> results/phase13_report.md")
