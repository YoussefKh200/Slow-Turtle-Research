"""Phase 9: exit research — trend capture vs risk reduction.

Entry fixed (22/55 cross, Monday open). Exits compared:
  ma_cross (book), atr_trail k*ATR14 (k=3,4,5) from highest close (chandelier),
  donchian_low n-week low (n=10,20), time stop 52w, vol stop (exit if ATR14/ATR52>1.5).
Stops are evaluated on weekly closes -> exit next open. Re-entry allowed while 22>55
whenever price makes a new 4-week high (else a stop-out would end the episode for good).
Optimize Return/Drawdown (Calmar), never return alone.
Run: python phase9_exits.py -> results/phase9_report.md
"""
import os
import numpy as np
import pandas as pd
from data import load_universe
from backtest import ma, atr
from stats import perf
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase9")

SKIP = 110


def run_exit(w: pd.DataFrame, mode: str, k: float = 0) -> pd.Series:
    c, o, idx = w["Close"], w["Open"], w.index
    sig = (ma(c, 22, "SMA") > ma(c, 55, "SMA")).fillna(False).values
    a14 = atr(w, 14).values
    vol_ratio = (atr(w, 14) / atr(w, 52)).values
    hi4 = c.rolling(4).max().values
    lo = {n: c.rolling(n).min().shift(1).values for n in (10, 20)}
    cv = c.values

    pos = np.zeros(len(w))
    in_pos, peak, entry_i = False, 0.0, 0
    for i in range(1, len(w)):
        if in_pos:
            peak = max(peak, cv[i])
            stop = not sig[i]  # every exit also honours the cross-down
            if mode == "atr_trail":
                stop = stop or cv[i] < peak - k * a14[i]
            elif mode == "donchian_low":
                stop = stop or cv[i] < lo[int(k)][i]
            elif mode == "time52":
                stop = stop or (i - entry_i) >= 52
            elif mode == "vol_stop":
                stop = stop or vol_ratio[i] > 1.5
            if stop:
                in_pos = False
        else:
            if sig[i] and (mode == "ma_cross" or cv[i] >= hi4[i]):
                in_pos, peak, entry_i = True, cv[i], i
                # decided at close i -> long from open i+1: handled by shift below
        pos[i] = in_pos
    ret_oo = o.pct_change().fillna(0.0)
    return (ret_oo * pd.Series(pos, index=idx).shift(2)).fillna(0.0)


if __name__ == "__main__":
    uni = load_universe()
    exits = [("MA cross (book)", "ma_cross", 0), ("ATR trail 3x", "atr_trail", 3),
             ("ATR trail 4x", "atr_trail", 4), ("ATR trail 5x", "atr_trail", 5),
             ("Donchian 10w low", "donchian_low", 10), ("Donchian 20w low", "donchian_low", 20),
             ("Time stop 52w", "time52", 0), ("Vol stop 1.5x", "vol_stop", 0)]
    rows = {}
    for label, mode, k in exits:
        stats = [perf(run_exit(w, mode, k), skip=SKIP) for w in uni.values()]
        d = pd.DataFrame(stats)
        rows[label] = {"median_sharpe": d["sharpe"].median(), "median_cagr": d["cagr"].median(),
                       "median_maxdd": d["max_dd"].median(), "median_calmar": d["calmar"].median()}
    df = pd.DataFrame(rows).T.sort_values("median_calmar", ascending=False)

    os.makedirs(OUT, exist_ok=True)
    lines = ["# Phase 9 — Exit Systems\n",
             "Entry fixed; exits compared. All stops also honour the MA cross-down.",
             "Re-entry on 4-week high while regime stays long. Ranked by **Calmar** (return/DD).\n",
             df.round(2).to_markdown(), ""]
    with open(os.path.join(OUT, "PHASE9_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(df.round(2).to_string())
    print("Report -> results/phase9/PHASE9_REPORT.md")
