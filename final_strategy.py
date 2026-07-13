"""
Slow Turtle — final validated strategy, standalone.

22/55-week SMA trend signal, per-sleeve EWMA vol targeting, equal-weight
16-asset portfolio, FTMO risk overlay (scale + drawdown-linear cut +
volatility brake). This is the exact configuration validated in
research phases 1-12 of github.com/YoussefKh200/Slow-Turtle-Research
(see RESEARCH.md there for the full derivation of every parameter below).

Backtested result (2007-2026 common window, no leverage, no costs):
    Sharpe 0.94   CAGR 2.4%   MaxDD -3.6%   Deflated Sharpe 0.875

No dependency on the rest of that repo — drop this file into any Python
environment with pandas/numpy/yfinance and run it.

    pip install pandas numpy yfinance
    python final_strategy.py
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------- config ---
# Every value here is a research finding, not a guess. Don't tune these
# without re-running the corresponding study (phase cited per block).

UNIVERSE = {   # name -> yfinance ticker.  phase6: diversification (0.36 avg
    "SPY": "SPY", "QQQ": "QQQ", "DIA": "DIA", "IWM": "IWM",           # sleeve corr) is the single biggest lever in the whole
    "NAS100": "^NDX", "SP500": "^GSPC",                               # program; FX sleeves are kept despite weak standalone
    "GOLD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F",                  # Sharpe because they diversify and vol-sizing bounds
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",    # their downside.
    "TECH": "XLK", "UTILITIES": "XLU", "INDUSTRIALS": "XLI", "MATERIALS": "XLB",
}

WEEKS_PER_YEAR = 52
FAST, SLOW = 22, 55            # phase3: flat plateau, 22/55 SMA at the 90th percentile already
SIGNAL_LAG_WEEKS = 2           # phase8: signal at week close -> position at next Monday's open
SLEEVE_VOL_TARGET = 0.10       # phase5: EWMA-26 10% target halves maxDD at unchanged Sharpe
EWMA_SPAN = 26
SLEEVE_CAP = 1.0               # no leverage per sleeve
PORTFOLIO_SCALE = 0.6          # phase11: scales ~10% sleeve vol down to ~6% portfolio vol for FTMO headroom
DD_FLOOR = 0.08                # phase11: exposure cut linearly to zero as drawdown approaches -8%
VOL_BRAKE_WINDOW = 8           # phase11: halve exposure when trailing 8w realized vol > 1.5x target
VOL_BRAKE_TRIGGER = 1.5
WARMUP_WEEKS = SLOW * 2        # let both MAs and the EWMA vol estimator stabilize

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ----------------------------------------------------------------- data ----
def get_weekly(ticker: str) -> pd.DataFrame:
    """Daily OHLC via yfinance (cached to data/<ticker>.csv) resampled to W-FRI."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, ticker + ".csv")  # matches data/ as committed by src/data.py
    if os.path.exists(cache):
        daily = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        daily = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)
        daily = daily[["Open", "High", "Low", "Close"]].dropna()
        daily.to_csv(cache)
    weekly = daily.resample("W-FRI").agg({"Open": "first", "High": "max",
                                          "Low": "min", "Close": "last"})
    return weekly.dropna()


# ------------------------------------------------------------- strategy ----
def sleeve_weight(weekly: pd.DataFrame) -> pd.Series:
    """Target weight for one sleeve: (trend signal) x (inverse-vol size), before lag."""
    c = weekly["Close"]
    signal = c.rolling(FAST).mean() > c.rolling(SLOW).mean()
    vol = c.pct_change().ewm(span=EWMA_SPAN, adjust=False).std() * np.sqrt(WEEKS_PER_YEAR)
    size = (SLEEVE_VOL_TARGET / vol).clip(upper=SLEEVE_CAP)
    return (signal * size).fillna(0.0)


def apply_overlay(port_ret: pd.Series) -> pd.Series:
    """Portfolio-level FTMO risk overlay, walked forward week by week.

    scale:      constant exposure multiplier (targets ~6% portfolio vol)
    dd_scaling: exposure *= max(0, 1 + drawdown/DD_FLOOR) -> hits 0 at -DD_FLOOR
    vol_brake:  halve exposure when trailing realized vol spikes past 1.5x target
    """
    rolling_vol = port_ret.rolling(VOL_BRAKE_WINDOW).std() * np.sqrt(WEEKS_PER_YEAR)
    out = np.zeros(len(port_ret))
    equity, peak = 1.0, 1.0
    for i, r in enumerate(port_ret.values):
        exposure = PORTFOLIO_SCALE
        drawdown = equity / peak - 1
        exposure *= max(0.0, 1 + drawdown / DD_FLOOR)
        if rolling_vol.iloc[i] > VOL_BRAKE_TRIGGER * SLEEVE_VOL_TARGET * PORTFOLIO_SCALE:
            exposure *= 0.5
        out[i] = exposure * r
        equity *= 1 + out[i]
        peak = max(peak, equity)
    return pd.Series(out, index=port_ret.index)


def backtest() -> dict:
    weights, opens = {}, {}
    for name, ticker in UNIVERSE.items():
        w = get_weekly(ticker)
        weights[name] = sleeve_weight(w)
        opens[name] = w["Open"]

    weights = pd.concat(weights, axis=1, sort=True).ffill().fillna(0.0)
    open_rets = pd.concat(opens, axis=1, sort=True).pct_change()

    # common window: only weeks where every sleeve has data
    df = open_rets.dropna()
    lagged_weights = (weights.shift(SIGNAL_LAG_WEEKS) / len(UNIVERSE)).reindex(df.index)
    portfolio_ret = (df * lagged_weights).sum(axis=1).iloc[WARMUP_WEEKS:]

    strategy_ret = apply_overlay(portfolio_ret)
    equity = (1 + strategy_ret).cumprod()
    years = len(strategy_ret) / WEEKS_PER_YEAR
    cagr = equity.iloc[-1] ** (1 / years) - 1
    sharpe = strategy_ret.mean() / strategy_ret.std() * np.sqrt(WEEKS_PER_YEAR)
    dd = (equity / equity.cummax() - 1)
    latest_targets = (weights.iloc[-1] * PORTFOLIO_SCALE / len(UNIVERSE)).rename("target_weight")

    return {"returns": strategy_ret, "equity": equity, "drawdown": dd,
            "sharpe": sharpe, "cagr": cagr, "max_dd": dd.min(),
            "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
            "latest_targets": latest_targets, "asof": str(df.index[-1].date())}


if __name__ == "__main__":
    out = backtest()
    print(f"Backtest {out['returns'].index[0].date()} -> {out['returns'].index[-1].date()} "
          f"({len(out['returns'])} weeks)")
    print(f"Sharpe {out['sharpe']:.2f}   CAGR {out['cagr']:.1%}   "
          f"MaxDD {out['max_dd']:.1%}   Calmar {out['calmar']:.2f}")

    out["equity"].to_csv("final_strategy_equity.csv", header=["equity"])
    print("equity curve -> final_strategy_equity.csv")

    print(f"\nCurrent target weights (asof {out['asof']}):")
    print(out["latest_targets"].round(4).to_string())
