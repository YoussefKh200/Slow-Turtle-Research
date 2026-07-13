"""Phase 3: question the 22/55 moving averages.

Grid: fast x slow x MA-type across the whole universe. Score = median Sharpe across
assets (common evaluation window). Prefer wide profitable plateaus over peaks.
Run: python phase3_ma_study.py -> results/phase3_report.md + phase3_heatmaps.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_universe
from backtest import ma, state_backtest
from stats import perf
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase3")
CHARTS = os.path.join(OUT, "charts")
os.makedirs(CHARTS, exist_ok=True)

FASTS = [10, 15, 20, 22, 30, 40, 50]
SLOWS = [40, 50, 55, 70, 100, 150]
KINDS = ["SMA", "EMA", "WMA", "HMA", "KAMA"]
SKIP = 155  # common warmup so every combo is judged on the same weeks


def run_grid() -> pd.DataFrame:
    uni = load_universe()
    # precompute all MAs once per (asset, kind, length)
    lengths = sorted(set(FASTS + SLOWS))
    mas = {(a, k, n): ma(w["Close"], n, k) for a, w in uni.items() for k in KINDS for n in lengths}
    rows = []
    for k in KINDS:
        for f in FASTS:
            for s in SLOWS:
                if f >= s:
                    continue
                sharpes = []
                for a, w in uni.items():
                    sig = mas[(a, k, f)] > mas[(a, k, s)]
                    p = perf(state_backtest(w, sig), skip=SKIP)
                    sharpes.append(p["sharpe"])
                sharpes = pd.Series(sharpes, index=list(uni))
                rows.append({"kind": k, "fast": f, "slow": s,
                             "median_sharpe": sharpes.median(),
                             "mean_sharpe": sharpes.mean(),
                             "pct_positive": (sharpes > 0).mean()})
    return pd.DataFrame(rows)


def heatmaps(df: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, len(KINDS), figsize=(4 * len(KINDS), 3.6), constrained_layout=True)
    vmax = df["median_sharpe"].abs().max()
    for ax, k in zip(axes, KINDS):
        sub = df[df["kind"] == k].pivot(index="fast", columns="slow", values="median_sharpe")
        im = ax.imshow(sub.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(sub.columns)), sub.columns)
        ax.set_yticks(range(len(sub.index)), sub.index)
        ax.set_title(f"{k} — median Sharpe", fontsize=10)
        ax.set_xlabel("slow"); ax.set_ylabel("fast")
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                v = sub.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.6 * vmax else "black")
    fig.colorbar(im, ax=axes, shrink=0.8, label="median Sharpe (16 assets)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stability(df: pd.DataFrame) -> pd.DataFrame:
    """Plateau score = mean Sharpe of grid neighbours; sensitivity = drop from best to neighbours."""
    out = []
    for k in KINDS:
        sub = df[df["kind"] == k].pivot(index="fast", columns="slow", values="median_sharpe")
        best = df[df["kind"] == k].nlargest(1, "median_sharpe").iloc[0]
        neigh = []
        fi, si = list(sub.index).index(best["fast"]), list(sub.columns).index(best["slow"])
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if 0 <= fi + di < len(sub.index) and 0 <= si + dj < len(sub.columns):
                    v = sub.values[fi + di, si + dj]
                    if not np.isnan(v):
                        neigh.append(v)
        out.append({"kind": k, "grid_median": sub.stack().median(), "grid_min": sub.stack().min(),
                    "grid_max": sub.stack().max(), "pct_grid_sharpe>0.3": (sub.stack() > 0.3).mean(),
                    "best": f"{int(best['fast'])}/{int(best['slow'])}",
                    "best_sharpe": best["median_sharpe"], "plateau_mean": np.mean(neigh)})
    return pd.DataFrame(out).set_index("kind")


if __name__ == "__main__":
    df = run_grid()
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "grid.csv"), index=False)
    heatmaps(df, os.path.join(CHARTS, "heatmaps.png"))
    stab = stability(df)

    base = df[(df["kind"] == "SMA") & (df["fast"] == 22) & (df["slow"] == 55)].iloc[0]
    rank = (df["median_sharpe"] < base["median_sharpe"]).mean()
    lines = [
        "# Phase 3 — Moving Average Study\n",
        f"Grid: fast {FASTS} x slow {SLOWS} x {KINDS}, 16 assets, common warmup {SKIP} weeks, no costs.",
        "Score = **median Sharpe across assets** (robust to one asset dominating).\n",
        f"Book baseline SMA 22/55: median Sharpe **{base['median_sharpe']:.2f}** — beats {rank:.0%} of all {len(df)} combos tested.\n",
        "## Stability by MA type\n",
        stab.round(2).to_markdown(), "",
        "![heatmaps](charts/heatmaps.png)\n",
        "## Median-Sharpe grids\n",
    ]
    for k in KINDS:
        sub = df[df["kind"] == k].pivot(index="fast", columns="slow", values="median_sharpe")
        lines += [f"### {k}\n", sub.round(2).to_markdown(), ""]
    top = df.nlargest(10, "median_sharpe")[["kind", "fast", "slow", "median_sharpe", "pct_positive"]]
    lines += ["## Top 10 combos (do NOT cherry-pick — look at plateaus)\n", top.round(2).to_markdown(index=False), ""]
    with open(os.path.join(OUT, "PHASE3_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(stab.round(2).to_string())
    print(f"\nBaseline 22/55 SMA percentile: {rank:.0%}")
    print("Report -> results/phase3/PHASE3_REPORT.md")
