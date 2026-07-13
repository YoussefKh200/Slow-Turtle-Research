"""Phase 8: entry execution — does anything beat 'just buy Monday open'?

Variants after a 22/55 cross signal (all exit on cross-down at next open, as baseline):
  immediate: next Monday open (book)
  close:     signal week's close (1 bar earlier information-wise; measures delay cost)
  confirm2:  wait 2 more weekly closes above -> then enter
  pullback3/5: wait for a 3%/5% dip below signal-week close; timeout 8w then market entry
  donchian4:  enter on break of 4-week high after signal
Metrics: median Sharpe, avg entry slippage vs immediate, % of trends missed.
Run: python phase8_entries.py -> results/phase8_report.md
"""
import os
import numpy as np
import pandas as pd
from common import load_universe, ma, perf
from slow_turtle import RESULTS_DIR

SKIP = 110
TIMEOUT = 8


def episodes(sig: pd.Series) -> list[tuple[int, int]]:
    """(signal_on_idx, signal_off_idx) pairs on integer positions."""
    d = sig.astype(int).diff()
    ons = np.where(d.values == 1)[0]
    offs = np.where(d.values == -1)[0]
    out = []
    for on in ons:
        off = next((o for o in offs if o > on), len(sig) - 1)
        out.append((on, off))
    return out


def entry_returns(w: pd.DataFrame, mode: str) -> pd.Series:
    """Weekly strategy returns with the given entry mode."""
    c, o = w["Close"], w["Open"]
    sig = (ma(c, 22, "SMA") > ma(c, 55, "SMA")).fillna(False)
    pos = pd.Series(0.0, index=w.index)
    for on, off in episodes(sig):
        start = None  # integer idx of first week we are long (from its open)
        if mode == "immediate":
            start = on + 1
        elif mode == "confirm2":
            start = on + 3 if off > on + 2 else None
        elif mode.startswith("pullback"):
            thresh = float(mode[-1]) / 100
            ref = c.iloc[on]
            for i in range(on + 1, min(on + 1 + TIMEOUT, off + 1)):
                if w["Low"].iloc[i] <= ref * (1 - thresh):
                    start = i + 1
                    break
            else:
                start = min(on + 1 + TIMEOUT, off)  # timeout -> market entry
        elif mode == "donchian4":
            hi4 = c.rolling(4).max()
            for i in range(on + 1, off + 1):
                if c.iloc[i] >= hi4.iloc[i]:
                    start = i + 1
                    break
        if start is None or start > off:
            continue
        # exit at open of week off+1 (signal died at close of off), matching baseline
        pos.iloc[start:min(off + 1, len(pos))] = 1.0
    ret_oo = o.pct_change().fillna(0.0)
    # pos[i]=1: long during week i (open i -> open i+1); that return lands at index i+1
    return (ret_oo * pos.shift(1)).fillna(0.0)


if __name__ == "__main__":
    uni = load_universe()
    modes = ["immediate", "confirm2", "pullback3", "pullback5", "donchian4"]
    rows = {}
    for mode in modes:
        stats = []
        for a, w in uni.items():
            p = perf(entry_returns(w, mode), skip=SKIP)
            stats.append(p)
        d = pd.DataFrame(stats)
        rows[mode] = {"median_sharpe": d["sharpe"].median(), "median_cagr": d["cagr"].median(),
                      "median_maxdd": d["max_dd"].median(), "median_calmar": d["calmar"].median()}
    df = pd.DataFrame(rows).T.sort_values("median_sharpe", ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = ["# Phase 8 — Entry Execution\n",
             "Same signal and exit; only the entry changes. Pullback entries timeout to market",
             f"after {TIMEOUT} weeks (else they miss the strongest trends entirely).\n",
             df.round(2).to_markdown(), ""]
    with open(os.path.join(RESULTS_DIR, "phase8_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(df.round(2).to_string())
    print("Report -> results/phase8_report.md")
