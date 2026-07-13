"""Shared research helpers: data, moving averages, signal backtest, performance stats."""
import numpy as np
import pandas as pd
from slow_turtle import UNIVERSE, fetch_daily, to_weekly, WEEKS_PER_YEAR

_cache: dict[str, pd.DataFrame] = {}


def load_universe() -> dict[str, pd.DataFrame]:
    if not _cache:
        for name, t in UNIVERSE.items():
            _cache[name] = to_weekly(fetch_daily(t))
    return _cache


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


def state_backtest(weekly: pd.DataFrame, signal: pd.Series) -> pd.Series:
    """Boolean signal at week close -> long from next week's open. Open-to-open returns."""
    ret_oo = weekly["Open"].pct_change()
    pos = signal.shift(2).fillna(False)
    return ret_oo.where(pos, 0.0).fillna(0.0)


def perf(r: pd.Series, skip: int = 0) -> dict:
    r = r.iloc[skip:]
    if len(r) < WEEKS_PER_YEAR or r.std() == 0:
        return {"sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan, "calmar": np.nan}
    eq = (1 + r).cumprod()
    years = len(r) / WEEKS_PER_YEAR
    cagr = eq.iloc[-1] ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    return {
        "sharpe": r.mean() / r.std() * np.sqrt(WEEKS_PER_YEAR),
        "cagr": cagr, "max_dd": dd,
        "calmar": cagr / abs(dd) if dd < 0 else np.nan,
    }


def atr(weekly: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = weekly["High"], weekly["Low"], weekly["Close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()
