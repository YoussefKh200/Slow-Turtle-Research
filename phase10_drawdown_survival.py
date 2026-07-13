"""Phase 10: drawdown survival — MAE/MFE, losing streaks, underwater time.

Per closed baseline trade: MAE (worst close vs entry while in trade), MFE (best),
consecutive-loss streaks, and portfolio-level underwater profile. Output feeds the
Phase 11 risk limits.
Run: python phase10_drawdown_survival.py -> results/phase10_report.md + phase10_mae_mfe.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_universe, ma, perf
from phase5_vol_sizing import sized_returns
from slow_turtle import RESULTS_DIR, backtest

SKIP = 110


def trade_excursions(w: pd.DataFrame) -> pd.DataFrame:
    _, trades, _ = backtest(w)
    if trades.empty:
        return trades
    c = w["Close"]
    mae, mfe = [], []
    for _, t in trades.iterrows():
        path = c.loc[t["entry_date"]:t["exit_date"]]
        mae.append(path.min() / t["entry_px"] - 1)
        mfe.append(path.max() / t["entry_px"] - 1)
    return trades.assign(mae=mae, mfe=mfe)


def max_streak(returns: pd.Series) -> int:
    streak = best = 0
    for r in returns:
        streak = streak + 1 if r <= 0 else 0
        best = max(best, streak)
    return best


if __name__ == "__main__":
    uni = load_universe()
    all_tr = pd.concat([trade_excursions(w).assign(asset=a) for a, w in uni.items()],
                       ignore_index=True)
    closed = all_tr[~all_tr["open"]]
    winners, losers = closed[closed["return"] > 0], closed[closed["return"] <= 0]

    streaks = {a: max_streak(trade_excursions(w)["return"]) for a, w in uni.items()}

    # portfolio underwater profile (Portfolio E, vol-targeted sleeves)
    rets = {a: sized_returns(w, "ewma", 0.10, 1).iloc[SKIP:] for a, w in uni.items()}
    port = pd.concat(rets, axis=1, sort=True).dropna().mean(axis=1)
    eq = (1 + port).cumprod()
    dd = eq / eq.cummax() - 1
    p = perf(port)

    # MAE/MFE scatter
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    ax.scatter(winners["mae"], winners["return"], s=22, c="#4269d0", label="winners", alpha=0.75)
    ax.scatter(losers["mae"], losers["return"], s=22, c="#ff725c", label="losers", alpha=0.75)
    ax.axvline(-0.10, color="#888", lw=1, ls="--")
    ax.annotate("-10% MAE", (-0.10, ax.get_ylim()[1]), fontsize=8, color="#666",
                xytext=(-52, -12), textcoords="offset points")
    ax.set_xlabel("MAE — worst excursion during trade")
    ax.set_ylabel("final trade return")
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("How much pain before the win? (all closed trades, 16 assets)", fontsize=11)
    fig.savefig(os.path.join(RESULTS_DIR, "phase10_mae_mfe.png"), dpi=150)
    plt.close(fig)

    q = lambda s, x: s.quantile(x)
    lines = [
        "# Phase 10 — Drawdown Survival (MAE / MFE)\n",
        f"{len(closed)} closed trades across 16 assets, baseline system.\n",
        "## Pain before gain\n",
        f"- Winners' MAE: median {q(winners['mae'], .5):.1%}, 75th pct {q(winners['mae'], .25):.1%}, worst {winners['mae'].min():.1%}",
        f"- Winners that first went >5% underwater: {(winners['mae'] < -0.05).mean():.0%}",
        f"- Winners that first went >10% underwater: {(winners['mae'] < -0.10).mean():.0%}",
        f"- Losers' MFE: median {q(losers['mfe'], .5):.1%} (they were green first — exits matter)",
        f"- MAE worse than -10% still ended positive: {(closed[closed['mae'] < -0.10]['return'] > 0).mean():.0%} of such trades\n",
        "![mae](phase10_mae_mfe.png)\n",
        "## Losing streaks (per asset, consecutive losing trades)\n",
        pd.Series(streaks).sort_values(ascending=False).to_frame("max_consecutive_losses").to_markdown(), "",
        "## Portfolio E underwater profile (vol-targeted, common window)\n",
        f"- Max drawdown: {p['max_dd']:.1%}",
        f"- Weeks underwater (dd<0): {(dd < 0).mean():.0%} of all weeks",
        f"- Weeks below -5%: {(dd < -0.05).mean():.0%}; below -10%: {(dd < -0.10).mean():.0%}",
        f"- Longest underwater spell: {int(max((dd < 0).astype(int).groupby((dd >= 0).cumsum()).sum()))} weeks\n",
        "## Implications for risk limits (Phase 11)",
        "- Trade-level stops tighter than ~10% MAE would kill a large share of eventual winners.",
        "- Risk must be managed at the **portfolio** level (sizing + diversification), not per trade.",
        f"- Expect 3-6 consecutive losing trades per asset; size so that streak stays inside the buffer.",
    ]
    with open(os.path.join(RESULTS_DIR, "phase10_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[2:12]))
    print("Report -> results/phase10_report.md")
