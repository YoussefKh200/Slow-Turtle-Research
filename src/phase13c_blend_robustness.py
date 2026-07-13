"""Phase 13c: robustness check on the Phase 13b TF+real-MR blend.

Phase 13b found TF 80% / MR 20% beats pure TF on both Sharpe (1.07 vs 0.94)
and MaxDD (-2.7% vs -3.6%) using the Mean-Reversion-Research program's real
sleeve. That was one blend-grid run on the full sample -- exactly the kind of
single number Phase 12 refused to trust for the TF-only config without a
walk-forward / bootstrap / DSR / cost-stress pass first. This applies the
same gauntlet to the blend.

MR's own leg was already stress-tested inside the Mean-Reversion-Research
program (its README: OOS Sharpe 0.55, survives 10bp slippage + 1-day delay,
DSR 0.81) -- not repeated here. What's new is whether the *combination* holds
up: out-of-sample window stability, bootstrap tail risk, DSR against the
combined search (TF's ~500 trials + Phase 13's 7 + Phase 13b's 7), and TF-side
cost/delay stress carried through the blend.

Run: python phase13c_blend_robustness.py -> results/phase13c/PHASE13C_REPORT.md
"""
import os
import numpy as np
import pandas as pd
from phase13b_real_mr_blend import load_real_mr_weekly  # must run before any TF import (see its docstring)

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase13c")
os.makedirs(OUT, exist_ok=True)
BLEND_TF_WEIGHT = 0.8  # phase13b standout
N_TRIALS = 500 + 7 + 7  # TF program (phase12) + phase13 grid + phase13b grid


def blend_with(tf: pd.Series, mr_weekly: pd.Series, wtf: float = BLEND_TF_WEIGHT) -> pd.Series:
    common = tf.index.intersection(mr_weekly.index)
    tf, mr = tf.loc[common], mr_weekly.loc[common]
    mr = mr * (tf.std() / mr.std())  # vol-match, same as phase13b
    return wtf * tf + (1 - wtf) * mr


if __name__ == "__main__":
    mr_weekly = load_real_mr_weekly()

    from stats import perf, sharpe
    from phase12_robustness import ftmo, portfolio, stressed_tf, deflated_sharpe, WEEKS_PER_YEAR

    tf = ftmo(portfolio())
    blend = blend_with(tf, mr_weekly)
    lines = ["# Phase 13c — Blend robustness (TF 80% / MR 20%)\n"]

    # 1. disjoint 5y windows
    win = 5 * WEEKS_PER_YEAR
    rows = {}
    for i in range(0, len(blend) - win + 1, win):
        chunk = blend.iloc[i:i + win]
        rows[f"{chunk.index[0].year}-{chunk.index[-1].year}"] = {
            "sharpe": sharpe(chunk), "cagr": perf(chunk)["cagr"], "max_dd": perf(chunk)["max_dd"]}
    wf = pd.DataFrame(rows).T
    lines += ["## Out-of-sample stability (disjoint 5y windows)\n", wf.round(2).to_markdown(), ""]

    # 2. Monte Carlo block bootstrap
    rng = np.random.default_rng(7)
    block, n_iter = 26, 2000
    vals = blend.values
    n_blocks = int(np.ceil(len(vals) / block))
    sims_dd, sims_sharpe = [], []
    for _ in range(n_iter):
        starts = rng.integers(0, len(vals) - block, n_blocks)
        sim = np.concatenate([vals[s:s + block] for s in starts])[:len(vals)]
        eq = np.cumprod(1 + sim)
        sims_dd.append((eq / np.maximum.accumulate(eq) - 1).min())
        sims_sharpe.append(sim.mean() / sim.std() * np.sqrt(WEEKS_PER_YEAR))
    lines += ["## Monte Carlo (26-week block bootstrap, 2000 paths)\n",
              f"- Sharpe: 5th pct {np.percentile(sims_sharpe, 5):.2f}, median {np.median(sims_sharpe):.2f}, "
              f"95th {np.percentile(sims_sharpe, 95):.2f}",
              f"- MaxDD: median {np.median(sims_dd):.1%}, 95th pct worst {np.percentile(sims_dd, 5):.1%}",
              f"- P(maxDD worse than -10%): {(np.array(sims_dd) < -0.10).mean():.1%}\n"]

    # 3. deflated Sharpe (combined search: TF's own DSR trial count + both blend grids)
    dsr = deflated_sharpe(blend, N_TRIALS)
    lines += ["## Deflated Sharpe Ratio\n",
              f"- Observed Sharpe {sharpe(blend):.2f} over {len(blend)} weeks, "
              f"deflated for {N_TRIALS} trials (TF's 500 + 7 phase13 blends + 7 phase13b blends):",
              f"- **DSR = {dsr:.3f}**\n"]

    # 4. TF-side cost/delay stress carried through the blend (MR side already
    #    stress-tested inside its own program -- see module docstring)
    srows = {}
    for name, tf_variant in {
        "clean": tf,
        "+1w execution delay": stressed_tf(delay=1),
        "10bp per turn": stressed_tf(cost_bp=10),
        "25bp per turn (spread stress)": stressed_tf(cost_bp=25),
    }.items():
        b = blend_with(tf_variant, mr_weekly)
        srows[name] = {"sharpe": sharpe(b), "cagr": perf(b)["cagr"], "max_dd": perf(b)["max_dd"]}
    sdf = pd.DataFrame(srows).T
    lines += ["## TF-side execution stress carried through the blend\n", sdf.round(2).to_markdown(), "",
              "Verdict: PASS if 5y windows stay positive, DSR > 0.9, stress Sharpe stays clear of the",
              "pure-TF baseline (0.94)."]

    with open(os.path.join(OUT, "PHASE13C_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(wf.round(2).to_string(), "\n")
    print(f"DSR = {dsr:.3f}\n")
    print(sdf.round(2).to_string())
    print("Report -> results/phase13c/PHASE13C_REPORT.md")
