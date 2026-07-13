"""Sanity checks for slow_turtle backtest logic on synthetic weekly data."""
import numpy as np
import pandas as pd
import slow_turtle as st


def make_weekly(closes):
    idx = pd.date_range("2000-01-07", periods=len(closes), freq="W-FRI")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": c.shift(1).fillna(c.iloc[0]), "High": c, "Low": c, "Close": c})


def test_uptrend_goes_long_and_wins():
    w = make_weekly(np.linspace(100, 300, 200))  # steady uptrend
    ret, trades, pos = st.backtest(w)
    assert pos.iloc[-1], "should be long at end of a persistent uptrend"
    assert (1 + ret).prod() > 1.5, "should capture most of the trend"
    assert len(trades) == 1 and trades.iloc[0]["open"]


def test_downtrend_stays_flat():
    w = make_weekly(np.linspace(300, 100, 200))
    ret, trades, pos = st.backtest(w)
    assert not pos.any(), "never long in a pure downtrend"
    assert (ret == 0).all() and trades.empty


def test_round_trip_trade_prices():
    # up for 120 weeks then down for 120 -> one closed trade at open prices
    closes = np.concatenate([np.linspace(100, 200, 120), np.linspace(200, 80, 120)])
    w = make_weekly(closes)
    _, trades, _ = st.backtest(w)
    closed = trades[~trades["open"]]
    assert len(closed) == 1
    t = closed.iloc[0]
    assert t["entry_px"] == w.loc[t["entry_date"], "Open"]
    assert t["exit_px"] == w.loc[t["exit_date"], "Open"]
    assert t["return"] > 0, "trend trade in an up-then-down series should still profit"


if __name__ == "__main__":
    test_uptrend_goes_long_and_wins()
    test_downtrend_stays_flat()
    test_round_trip_trade_prices()
    print("all sanity checks passed")
