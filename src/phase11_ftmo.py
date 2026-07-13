"""Phase 11: FTMO/prop-firm adaptation of the diversified system.

Base = Portfolio E (16 sleeves, EWMA vol target per sleeve, cap 1x, equal weight).
Overlays tested:
  target scaling: portfolio vol target 10% -> 6% / 4% (scale all weights)
  dd_scaling:     exposure *= max(0, 1 - dd/8%)  (linear cut, flat at -8%)
  vol_brake:      halve exposure when realized 8w portfolio vol > 1.5x target
Checks: max total DD < 10% (prefer 5-8%), worst week, weekly-loss tails.
Run: python phase11_ftmo.py -> results/phase11_report.md + phase11_equity.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_universe
from backtest import WEEKS_PER_YEAR
from stats import perf
from phase5_vol_sizing import sized_returns
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase11")
CHARTS = os.path.join(OUT, "charts")
os.makedirs(CHARTS, exist_ok=True)

SKIP = 110
DD_FLOOR = 0.08  # exposure reaches 0 at -8% drawdown


def base_portfolio() -> pd.Series:
    uni = load_universe()
    rets = {a: sized_returns(w, "ewma", 0.10, 1).iloc[SKIP:] for a, w in uni.items()}
    return pd.concat(rets, axis=1, sort=True).dropna().mean(axis=1)


def overlay(port: pd.Series, scale: float, dd_scaling: bool, vol_brake: bool,
            target: float = 0.10) -> pd.Series:
    """Walk forward week by week applying exposure rules to the base return stream."""
    out = np.zeros(len(port))
    eq, peak = 1.0, 1.0
    rv = port.rolling(8).std() * np.sqrt(WEEKS_PER_YEAR)
    for i, r in enumerate(port.values):
        expo = scale
        dd = eq / peak - 1
        if dd_scaling:
            expo *= max(0.0, 1 + dd / DD_FLOOR)  # dd negative -> linear cut
        if vol_brake and rv.iloc[i] > 1.5 * target * scale:
            expo *= 0.5
        out[i] = expo * r
        eq *= 1 + out[i]
        peak = max(peak, eq)
    return pd.Series(out, index=port.index)


def stats(r: pd.Series) -> dict:
    p = perf(r)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {"cagr": p["cagr"], "sharpe": p["sharpe"], "max_dd": p["max_dd"],
            "calmar": p["calmar"], "worst_week": r.min(),
            "pct_weeks<-2%": (r < -0.02).mean(),
            "ftmo_pass(dd<10%)": "YES" if p["max_dd"] > -0.10 else "no"}


if __name__ == "__main__":
    port = base_portfolio()
    configs = {
        "raw Portfolio E (10% vol)": overlay(port, 1.0, False, False),
        "scaled to ~6% vol": overlay(port, 0.6, False, False),
        "scaled ~6% + dd-scaling": overlay(port, 0.6, True, False),
        "scaled ~6% + dd + vol-brake": overlay(port, 0.6, True, True),
        "scaled to ~4% vol + dd": overlay(port, 0.4, True, False),
    }
    df = pd.DataFrame({k: stats(v) for k, v in configs.items()}).T

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#9c6b4e"]
    for (name, r), col in zip(configs.items(), colors):
        eq = (1 + r).cumprod()
        ax.plot(eq.index, eq.values, lw=2, color=col, label=name)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("FTMO overlays on Portfolio E", fontsize=11)
    fig.savefig(os.path.join(CHARTS, "equity.png"), dpi=150)
    plt.close(fig)

    lines = ["# Phase 11 — FTMO Adaptation\n",
             f"Base: Portfolio E. Overlays: exposure scaling, drawdown-linear cut (flat at -{DD_FLOOR:.0%}),",
             "8-week realized-vol brake. Weekly system: daily-loss risk is bounded by weekly tails shown.\n",
             df.round(3).to_markdown(), "",
             "![equity](charts/equity.png)\n",
             "Notes:",
             "- The dd-scaling overlay guarantees exposure hits 0 before the FTMO 10% total-loss line.",
             "- Daily 5% loss limit: portfolio vol ~6%/yr => weekly sigma ~0.8%; a 5% daily move at",
             "  these exposures is a >5-sigma event; per-asset caps (1x) bound gap risk.",
             ]
    with open(os.path.join(OUT, "PHASE11_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(df.round(3).to_string())
    print("Report -> results/phase11/PHASE11_REPORT.md")
