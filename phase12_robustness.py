"""Phase 12: robustness — walk-forward, Monte Carlo, deflated Sharpe, perturbation.

System under test: FTMO config (Portfolio E sleeves, 6% scaling + dd + vol-brake).
  1. Walk-forward: no parameters are fit, so WF = out-of-sample stability of Sharpe
     across disjoint 5y windows.
  2. Monte Carlo on weekly portfolio returns: block bootstrap (26w blocks), plus
     execution stress on sleeves: 1-week delay, 10bp/turn costs, 25bp spread stress.
  3. Deflated Sharpe: correct the observed Sharpe for the number of configs tried
     across the whole research program (~500 variants).
  4. Perturbation: FAST/SLOW +-20%, vol-target +-20%.
Run: python phase12_robustness.py -> results/phase12_report.md
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as sps
from common import load_universe, ma, perf, WEEKS_PER_YEAR
from phase5_vol_sizing import sized_returns
from phase11_ftmo import overlay
from slow_turtle import RESULTS_DIR

SKIP = 110
N_TRIALS = 500  # configs examined across phases 3-11 (honest upper-ish bound)


def portfolio(fast=22, slow=55, target=0.10) -> pd.Series:
    uni = load_universe()
    rets = {}
    for a, w in uni.items():
        c = w["Close"]
        sig = ma(c, fast, "SMA") > ma(c, slow, "SMA")
        vol = c.pct_change().ewm(span=26, adjust=False).std() * np.sqrt(WEEKS_PER_YEAR)
        weight = (target / vol).clip(upper=1)
        pos = (sig * weight).shift(2).fillna(0.0)
        rets[a] = (w["Open"].pct_change().fillna(0.0) * pos).iloc[SKIP:]
    return pd.concat(rets, axis=1, sort=True).dropna().mean(axis=1)


def ftmo(port: pd.Series) -> pd.Series:
    return overlay(port, 0.6, True, True)


def sharpe(r: pd.Series) -> float:
    return r.mean() / r.std() * np.sqrt(WEEKS_PER_YEAR) if r.std() > 0 else np.nan


def deflated_sharpe(r: pd.Series, n_trials: int) -> float:
    """Bailey & Lopez de Prado DSR: prob. that true Sharpe > 0 given selection bias."""
    sr = sharpe(r) / np.sqrt(WEEKS_PER_YEAR)  # per-period
    n = len(r)
    skew, kurt = sps.skew(r), sps.kurtosis(r, fisher=False)
    # expected max Sharpe of n_trials pure-noise strategies
    emc = 0.5772156649
    var_sr = 1.0 / n
    sr0 = np.sqrt(var_sr) * ((1 - emc) * sps.norm.ppf(1 - 1 / n_trials)
                             + emc * sps.norm.ppf(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1))
    return float(sps.norm.cdf((sr - sr0) / denom))


if __name__ == "__main__":
    port = portfolio()
    strat = ftmo(port)
    lines = ["# Phase 12 — Robustness Testing\n"]

    # 1. disjoint 5-year windows
    win = 5 * WEEKS_PER_YEAR
    rows = {}
    for i in range(0, len(strat) - win + 1, win):
        chunk = strat.iloc[i:i + win]
        rows[f"{chunk.index[0].year}-{chunk.index[-1].year}"] = {
            "sharpe": sharpe(chunk), "cagr": perf(chunk)["cagr"], "max_dd": perf(chunk)["max_dd"]}
    wf = pd.DataFrame(rows).T
    lines += ["## Out-of-sample stability (disjoint 5y windows, no fitting anywhere)\n",
              wf.round(2).to_markdown(), ""]

    # 2. Monte Carlo block bootstrap
    rng = np.random.default_rng(7)
    block = 26
    n_iter = 2000
    vals = strat.values
    sims_dd, sims_sharpe = [], []
    n_blocks = int(np.ceil(len(vals) / block))
    for _ in range(n_iter):
        starts = rng.integers(0, len(vals) - block, n_blocks)
        sim = np.concatenate([vals[s:s + block] for s in starts])[:len(vals)]
        eq = np.cumprod(1 + sim)
        sims_dd.append((eq / np.maximum.accumulate(eq) - 1).min())
        sims_sharpe.append(sim.mean() / sim.std() * np.sqrt(WEEKS_PER_YEAR))
    lines += ["## Monte Carlo (26-week block bootstrap, 2000 paths)\n",
              f"- Sharpe: 5th pct {np.percentile(sims_sharpe, 5):.2f}, median {np.median(sims_sharpe):.2f}, 95th {np.percentile(sims_sharpe, 95):.2f}",
              f"- MaxDD: median {np.median(sims_dd):.1%}, 95th pct worst {np.percentile(sims_dd, 5):.1%}",
              f"- P(maxDD worse than -10%): {(np.array(sims_dd) < -0.10).mean():.1%}\n"]

    # execution stress on the sleeve level
    uni = load_universe()
    def stressed(delay=0, cost_bp=0):
        """Identical to portfolio() except extra signal delay and per-turn costs."""
        rets = {}
        for a, w in uni.items():
            c = w["Close"]
            sig = ma(c, 22, "SMA") > ma(c, 55, "SMA")
            vol = c.pct_change().ewm(span=26, adjust=False).std() * np.sqrt(WEEKS_PER_YEAR)
            pos = (sig * (0.10 / vol).clip(upper=1)).shift(2 + delay).fillna(0.0)
            turn = pos.diff().abs().fillna(0.0)
            rets[a] = (w["Open"].pct_change().fillna(0.0) * pos - turn * cost_bp / 1e4).iloc[SKIP:]
        return ftmo(pd.concat(rets, axis=1, sort=True).dropna().mean(axis=1))

    stress = {
        "clean": strat,
        "+1w execution delay": stressed(delay=1),
        "10bp per turn": stressed(cost_bp=10),
        "25bp per turn (spread stress)": stressed(cost_bp=25),
        "delay + 25bp": stressed(delay=1, cost_bp=25),
    }
    sdf = pd.DataFrame({k: {"sharpe": sharpe(v), "cagr": perf(v)["cagr"],
                            "max_dd": perf(v)["max_dd"]} for k, v in stress.items()}).T
    lines += ["## Execution stress\n", sdf.round(2).to_markdown(), ""]

    # 3. deflated Sharpe
    dsr = deflated_sharpe(strat, N_TRIALS)
    lines += ["## Deflated Sharpe Ratio\n",
              f"- Observed Sharpe {sharpe(strat):.2f} over {len(strat)} weeks, deflated for {N_TRIALS} trials:",
              f"- **DSR = {dsr:.3f}** (probability the true Sharpe > 0 after selection bias)\n"]

    # 4. parameter perturbation
    prows = {}
    for f, s, t in [(22, 55, .10), (18, 44, .10), (26, 66, .10), (18, 66, .10),
                    (26, 44, .10), (22, 55, .08), (22, 55, .12)]:
        r = ftmo(portfolio(f, s, t))
        prows[f"fast={f} slow={s} target={t:.0%}"] = {
            "sharpe": sharpe(r), "cagr": perf(r)["cagr"], "max_dd": perf(r)["max_dd"]}
    pdf = pd.DataFrame(prows).T
    lines += ["## Parameter perturbation (+-20%)\n", pdf.round(2).to_markdown(), "",
              "Verdict: PASS if 5y windows stay positive, DSR > 0.9, stress Sharpe > 0.5,",
              "and perturbed Sharpe stays within ~0.15 of base."]

    with open(os.path.join(RESULTS_DIR, "phase12_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(wf.round(2).to_string(), "\n")
    print(sdf.round(2).to_string(), "\n")
    print(f"DSR = {dsr:.3f}")
    print(pdf.round(2).to_string())
    print("Report -> results/phase12_report.md")
