"""Performance statistics shared by every phase."""
import numpy as np
import pandas as pd
from backtest import WEEKS_PER_YEAR


def sharpe(r: pd.Series) -> float:
    return r.mean() / r.std() * np.sqrt(WEEKS_PER_YEAR) if r.std() > 0 else np.nan


def perf(r: pd.Series, skip: int = 0) -> dict:
    r = r.iloc[skip:]
    if len(r) < WEEKS_PER_YEAR or r.std() == 0:
        return {"sharpe": np.nan, "cagr": np.nan, "max_dd": np.nan, "calmar": np.nan}
    eq = (1 + r).cumprod()
    years = len(r) / WEEKS_PER_YEAR
    cagr = eq.iloc[-1] ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    return {
        "sharpe": sharpe(r), "cagr": cagr, "max_dd": dd,
        "calmar": cagr / abs(dd) if dd < 0 else np.nan,
    }


def drawdown_stats(equity: pd.Series) -> dict:
    peak = equity.cummax()
    dd = equity / peak - 1
    underwater = dd < 0
    spells, run = [], 0
    for u in underwater:
        run = run + 1 if u else 0
        if run:
            spells.append(run)
    return {"max_dd": dd.min(), "longest_underwater_weeks": max(spells) if spells else 0,
            "dd_series": dd}


def full_metrics(strat_ret: pd.Series, trades: pd.DataFrame, pos: pd.Series,
                 warmup: int = 56) -> dict:
    """The Phase 2 metric battery for one asset."""
    valid = strat_ret.index[warmup:]
    r = strat_ret.loc[valid]
    if len(r) < WEEKS_PER_YEAR:
        return {}
    equity = (1 + r).cumprod()
    years = len(r) / WEEKS_PER_YEAR
    dd = drawdown_stats(equity)
    downside = r[r < 0].std() * np.sqrt(WEEKS_PER_YEAR)
    t = trades
    wins, losses = t[t["return"] > 0], t[t["return"] <= 0]
    gross_win, gross_loss = wins["return"].sum(), abs(losses["return"].sum())
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    return {
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "years": round(years, 1),
        "total_return": equity.iloc[-1] - 1, "cagr": cagr,
        "ann_vol": r.std() * np.sqrt(WEEKS_PER_YEAR),
        "sharpe": sharpe(r),
        "sortino": r.mean() * WEEKS_PER_YEAR / downside if downside > 0 else np.nan,
        "calmar": cagr / abs(dd["max_dd"]) if dd["max_dd"] < 0 else np.nan,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else np.inf,
        "win_rate": len(wins) / len(t) if len(t) else np.nan,
        "avg_winner": wins["return"].mean() if len(wins) else np.nan,
        "avg_loser": losses["return"].mean() if len(losses) else np.nan,
        "largest_winner": t["return"].max() if len(t) else np.nan,
        "largest_loser": t["return"].min() if len(t) else np.nan,
        "max_dd": dd["max_dd"],
        "longest_underwater_yrs": dd["longest_underwater_weeks"] / WEEKS_PER_YEAR,
        "exposure": pos.loc[valid].mean(),
        "n_trades": len(t),
        "avg_weeks_held": t["weeks_held"].mean() if len(t) else np.nan,
        "negative_years": int((yearly < 0).sum()), "total_years_counted": len(yearly),
        "yearly": yearly, "equity": equity, "trades": t,
    }


def profit_concentration(trades: pd.DataFrame) -> dict:
    """What share of trades produces the profits? (sum of simple returns as proxy)"""
    if trades.empty:
        return {}
    r = trades["return"].sort_values(ascending=False)
    net = r.sum()
    if net <= 0:
        return {"note": "net unprofitable"}
    cum = r.cumsum()
    return {
        "top10pct_share": r.head(max(1, len(r) // 10)).sum() / net,
        "n_trades_for_100pct": min(int((cum < net).sum()) + 1, len(r)),
        "n_trades": len(r),
    }
