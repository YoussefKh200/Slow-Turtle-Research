"""Data loading with local CSV cache."""
import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

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

_cache: dict[str, pd.DataFrame] = {}


def get_data(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Return daily OHLC for ticker, cached to data/<ticker>.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if not refresh and os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    raw = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    raw.columns = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
    raw = raw[["Open", "High", "Low", "Close"]].dropna()
    raw.to_csv(path)
    return raw


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    w = daily.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    return w.dropna()


def load_universe(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Weekly OHLC for every universe asset, memoized per process."""
    if not _cache or refresh:
        for name, t in UNIVERSE.items():
            _cache[name] = to_weekly(get_data(t, refresh=refresh))
    return _cache


if __name__ == "__main__":
    for name, w in load_universe().items():
        print(f"{name:12s} {w.index[0].date()} -> {w.index[-1].date()}  {len(w)} weeks")
