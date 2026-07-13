"""Phase 1+2: Slow Turtle baseline (22/55-week SMA cross, long only, Monday-open execution).

Signal at weekly close -> position from next week's open. Open-to-open weekly returns.
Run: python slow_turtle.py   -> results/baseline_report.md, results/trades.csv, results/metrics.csv
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

FAST, SLOW = 22, 55
WEEKS_PER_YEAR = 52

UNIVERSE = {
    # Indices
    "SPY": "SPY", "QQQ": "QQQ", "DIA": "DIA", "IWM": "IWM",
    "NAS100": "^NDX", "SP500": "^GSPC",
    # Commodities (continuous futures)
    "GOLD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F",
    # Currencies
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    # Stock groups via sector ETFs (avoids survivorship bias of picking names)
    "TECH": "XLK", "UTILITIES": "XLU", "INDUSTRIALS": "XLI", "MATERIALS": "XLB",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def fetch_daily(ticker: str) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, ticker.replace("^", "_").replace("=", "_") + ".csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.to_csv(cache)
    return df


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    w = daily.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    return w.dropna()


def backtest(weekly: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Returns (weekly strategy returns, trade log, position series)."""
    close, open_ = weekly["Close"], weekly["Open"]
    signal = close.rolling(FAST).mean() > close.rolling(SLOW).mean()  # state at week close
    # position during window (open[t-1] -> open[t]) is decided at close of week t-2
    ret_oo = open_.pct_change()
    pos = signal.shift(2).fillna(False)
    strat_ret = ret_oo.where(pos, 0.0).fillna(0.0)

    # trade log: signal flip at close of week t -> executed at open of week t+1
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
        ex_i = idx.get_loc(ex_candidates[0]) + 1 if ex_candidates else len(idx) - 1
        ex_i = min(ex_i, len(idx) - 1)
        entry_px, exit_px = open_.iloc[ent_i], open_.iloc[ex_i]
        trades.append({
            "entry_date": idx[ent_i], "exit_date": idx[ex_i],
            "entry_px": entry_px, "exit_px": exit_px,
            "return": exit_px / entry_px - 1,
            "weeks_held": ex_i - ent_i,
            "open": not ex_candidates,
        })
    return strat_ret, pd.DataFrame(trades), pos


def drawdown_stats(equity: pd.Series) -> dict:
    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = dd.min()
    # longest underwater spell (weeks)
    underwater = dd < 0
    spells, run = [], 0
    for u in underwater:
        run = run + 1 if u else 0
        if run:
            spells.append(run)
    longest = max(spells) if spells else 0
    return {"max_dd": max_dd, "longest_underwater_weeks": longest, "dd_series": dd}


def metrics(strat_ret: pd.Series, trades: pd.DataFrame, pos: pd.Series) -> dict:
    # trim to period where SLOW SMA exists
    valid = strat_ret.index[SLOW + 1:]
    r = strat_ret.loc[valid]
    if len(r) < WEEKS_PER_YEAR:
        return {}
    equity = (1 + r).cumprod()
    years = len(r) / WEEKS_PER_YEAR
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(WEEKS_PER_YEAR)
    sharpe = (r.mean() / r.std() * np.sqrt(WEEKS_PER_YEAR)) if r.std() > 0 else 0.0
    downside = r[r < 0].std() * np.sqrt(WEEKS_PER_YEAR)
    sortino = (r.mean() * WEEKS_PER_YEAR / downside) if downside > 0 else np.nan
    dd = drawdown_stats(equity)
    calmar = cagr / abs(dd["max_dd"]) if dd["max_dd"] < 0 else np.nan

    t = trades
    wins, losses = t[t["return"] > 0], t[t["return"] <= 0]
    gross_win, gross_loss = wins["return"].sum(), abs(losses["return"].sum())
    yearly = (1 + r).groupby(r.index.year).prod() - 1

    return {
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "years": round(years, 1),
        "total_return": total, "cagr": cagr, "ann_vol": vol,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
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
    n_for_all = int((cum < net).sum()) + 1  # top-N trades covering 100% of net profit
    return {
        "top10pct_share": r.head(max(1, len(r) // 10)).sum() / net,
        "n_trades_for_100pct": min(n_for_all, len(r)),
        "n_trades": len(r),
    }


def run_all() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, all_trades, extras = [], [], {}
    for name, ticker in UNIVERSE.items():
        try:
            weekly = to_weekly(fetch_daily(ticker))
        except Exception as e:  # ticker download can fail; report and continue
            print(f"  SKIP {name} ({ticker}): {e}")
            continue
        strat_ret, trades, pos = backtest(weekly)
        m = metrics(strat_ret, trades, pos)
        if not m:
            print(f"  SKIP {name}: insufficient history")
            continue
        extras[name] = {"yearly": m.pop("yearly"), "equity": m.pop("equity"),
                        "concentration": profit_concentration(m.pop("trades"))}
        rows.append({"asset": name, "ticker": ticker, **m})
        trades = trades.assign(asset=name)
        all_trades.append(trades)
        print(f"  {name}: CAGR {m['cagr']:.1%}, Sharpe {m['sharpe']:.2f}, MaxDD {m['max_dd']:.1%}, {m['n_trades']} trades")
    return pd.DataFrame(rows).set_index("asset"), pd.concat(all_trades, ignore_index=True), extras


def write_report(df: pd.DataFrame, trades: pd.DataFrame, extras: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"))
    trades.to_csv(os.path.join(RESULTS_DIR, "trades.csv"), index=False)

    pct = lambda x: f"{x:.1%}" if pd.notna(x) else "-"
    num = lambda x: f"{x:.2f}" if pd.notna(x) else "-"
    lines = [
        "# Slow Turtle Baseline — Phase 1+2 Report",
        f"\nSystem: {FAST}/{SLOW}-week SMA cross, weekly candles, long only, next-Monday-open execution. No costs.\n",
        "## Per-asset performance\n",
        "| Asset | Period | CAGR | Vol | Sharpe | Sortino | Calmar | MaxDD | PF | WinRate | Trades | AvgHold(w) | Exposure | Neg.Yrs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a, r in df.iterrows():
        lines.append(
            f"| {a} | {r['start']}→{r['end']} | {pct(r['cagr'])} | {pct(r['ann_vol'])} | {num(r['sharpe'])} "
            f"| {num(r['sortino'])} | {num(r['calmar'])} | {pct(r['max_dd'])} | {num(r['profit_factor'])} "
            f"| {pct(r['win_rate'])} | {int(r['n_trades'])} | {num(r['avg_weeks_held'])} | {pct(r['exposure'])} "
            f"| {int(r['negative_years'])}/{int(r['total_years_counted'])} |")

    lines += ["\n## Trade P&L structure\n",
              "| Asset | AvgWin | AvgLoss | LargestWin | LargestLoss | Top10% trades = %profit | Trades for 100% profit | Underwater max (yrs) |",
              "|---|---|---|---|---|---|---|---|"]
    for a, r in df.iterrows():
        c = extras[a]["concentration"]
        share = pct(c.get("top10pct_share")) if "top10pct_share" in c else c.get("note", "-")
        nfull = f"{c['n_trades_for_100pct']}/{c['n_trades']}" if "n_trades_for_100pct" in c else "-"
        lines.append(f"| {a} | {pct(r['avg_winner'])} | {pct(r['avg_loser'])} | {pct(r['largest_winner'])} "
                     f"| {pct(r['largest_loser'])} | {share} | {nfull} | {num(r['longest_underwater_yrs'])} |")

    lines += ["\n## Yearly returns (strategy)\n"]
    yearly = pd.DataFrame({a: e["yearly"] for a, e in extras.items()})
    lines.append(yearly.map(lambda x: f"{x:.1%}" if pd.notna(x) else "").to_markdown())

    with open(os.path.join(RESULTS_DIR, "baseline_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport -> {os.path.join(RESULTS_DIR, 'baseline_report.md')}")


if __name__ == "__main__":
    df, trades, extras = run_all()
    write_report(df, trades, extras)
