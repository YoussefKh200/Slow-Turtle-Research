"""Moving averages, indicators and the weekly signal backtest.

Execution convention everywhere: signal at weekly close -> position from the
NEXT week's Monday open. With open-to-open returns that is a 2-week shift.
"""
import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


# ---------- moving averages ----------
def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def kama(s: pd.Series, n: int, fast: int = 2, slow: int = 30) -> pd.Series:
    change = (s - s.shift(n)).abs()
    vol = s.diff().abs().rolling(n).sum()
    er = (change / vol).fillna(0.0)
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.full(len(s), np.nan)
    vals, scs = s.to_numpy(dtype=float), sc.to_numpy(dtype=float)
    prev = vals[0]
    for i in range(len(vals)):
        prev = prev + scs[i] * (vals[i] - prev)
        out[i] = prev
    return pd.Series(out, index=s.index)


def ma(s: pd.Series, n: int, kind: str) -> pd.Series:
    if kind == "SMA":
        return s.rolling(n).mean()
    if kind == "EMA":
        return s.ewm(span=n, adjust=False).mean()
    if kind == "WMA":
        return wma(s, n)
    if kind == "HMA":
        return wma(2 * wma(s, n // 2) - wma(s, n), max(2, int(np.sqrt(n))))
    if kind == "KAMA":
        return kama(s, n)
    raise ValueError(kind)


# ---------- indicators ----------
def atr(weekly: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = weekly["High"], weekly["Low"], weekly["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ---------- backtest ----------
def state_backtest(weekly: pd.DataFrame, signal: pd.Series) -> pd.Series:
    """Boolean signal at week close -> long from next week's open. Open-to-open returns."""
    ret_oo = weekly["Open"].pct_change()
    pos = signal.shift(2).fillna(False)
    return ret_oo.where(pos, 0.0).fillna(0.0)


def cross_backtest(weekly: pd.DataFrame, fast: int = 22, slow: int = 55,
                   kind: str = "SMA") -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """MA-cross system with trade log. Returns (weekly returns, trades, position)."""
    close, open_ = weekly["Close"], weekly["Open"]
    signal = ma(close, fast, kind) > ma(close, slow, kind)
    strat_ret = state_backtest(weekly, signal)
    pos = signal.shift(2).fillna(False)

    flips = signal.astype(int).diff()
    entries = list(signal.index[(flips == 1).values])
    exits = list(signal.index[(flips == -1).values])
    trades = []
    idx = weekly.index
    for ent in entries:
        ent_i = idx.get_loc(ent) + 1
        if ent_i >= len(idx):
            continue
        ex_candidates = [e for e in exits if e > ent]
        ex_i = min(idx.get_loc(ex_candidates[0]) + 1 if ex_candidates else len(idx) - 1,
                   len(idx) - 1)
        entry_px, exit_px = open_.iloc[ent_i], open_.iloc[ex_i]
        trades.append({
            "entry_date": idx[ent_i], "exit_date": idx[ex_i],
            "entry_px": entry_px, "exit_px": exit_px,
            "return": exit_px / entry_px - 1,
            "weeks_held": ex_i - ent_i,
            "open": not ex_candidates,
        })
    return strat_ret, pd.DataFrame(trades), pos
