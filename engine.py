"""Production engine — the system the research (Phases 1-13) validated.

Pipeline: data -> weekly features -> trend signal (22/55 SMA) -> vol-targeted sizing
-> equal-weight portfolio -> FTMO risk overlays -> target weights + orders.

Every parameter lives in CONFIG (override via a JSON file: python engine.py my.json).
Every run is saved to runs/<timestamp>/ (config, weights, equity, orders) and the
latest target weights land in runs/latest_signals.csv for the MT5 bridge EA to read.
Run: python engine.py [config.json]
"""
import json
import os
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from common import load_universe, ma, perf, WEEKS_PER_YEAR
from phase11_ftmo import overlay
from slow_turtle import UNIVERSE

CONFIG = {
    "universe": list(UNIVERSE),          # names from slow_turtle.UNIVERSE
    "fast": 22, "slow": 55, "ma_kind": "SMA",
    "sleeve_vol_target": 0.10,           # per-asset EWMA vol target
    "sleeve_cap": 1.0,                   # no leverage per sleeve
    "ewma_span": 26,
    "signal_lag_weeks": 2,               # signal at close -> position at next Monday open
    "portfolio_scale": 0.6,              # Phase 11: ~6% portfolio vol
    "dd_scaling": True,                  # linear exposure cut, flat at dd_floor
    "vol_brake": True,                   # halve exposure when 8w vol > 1.5x target
    "warmup_weeks": 110,
}


def sleeve_weights(w: pd.DataFrame, cfg: dict) -> pd.Series:
    """Target weight series for one asset (before portfolio overlays)."""
    c = w["Close"]
    sig = ma(c, cfg["fast"], cfg["ma_kind"]) > ma(c, cfg["slow"], cfg["ma_kind"])
    vol = c.pct_change().ewm(span=cfg["ewma_span"], adjust=False).std() * np.sqrt(WEEKS_PER_YEAR)
    weight = (cfg["sleeve_vol_target"] / vol).clip(upper=cfg["sleeve_cap"])
    return (sig * weight).fillna(0.0)


def run(cfg: dict) -> dict:
    uni = {a: w for a, w in load_universe().items() if a in cfg["universe"]}
    n = len(uni)
    weights = pd.concat({a: sleeve_weights(w, cfg) for a, w in uni.items()},
                        axis=1, sort=True).ffill().fillna(0.0) / n
    # common window only: judge the portfolio where every asset actually trades
    rets = pd.concat({a: w["Open"].pct_change() for a, w in uni.items()},
                     axis=1, sort=True).dropna()
    lagged = weights.shift(cfg["signal_lag_weeks"]).reindex(rets.index)
    port = (rets * lagged).sum(axis=1).iloc[cfg["warmup_weeks"]:]
    strat = overlay(port, cfg["portfolio_scale"], cfg["dd_scaling"], cfg["vol_brake"],
                    target=cfg["sleeve_vol_target"])
    # current live targets = latest weights x current overlay exposure ratio
    expo = strat.iloc[-1] / port.iloc[-1] if port.iloc[-1] != 0 else cfg["portfolio_scale"]
    targets = (weights.iloc[-1] * expo).rename("target_weight")
    return {"stats": perf(strat), "equity": (1 + strat).cumprod(),
            "targets": targets, "asof": str(weights.index[-1].date())}


def save_run(out: dict, cfg: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rundir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", stamp)
    os.makedirs(rundir, exist_ok=True)
    with open(os.path.join(rundir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(rundir, "stats.json"), "w") as f:
        json.dump({k: float(v) for k, v in out["stats"].items()}, f, indent=2)
    out["equity"].to_csv(os.path.join(rundir, "equity.csv"))
    sig = out["targets"].to_frame()
    sig["ticker"] = [UNIVERSE[a] for a in sig.index]
    sig["asof"] = out["asof"]
    sig.to_csv(os.path.join(rundir, "signals.csv"))
    sig.to_csv(os.path.join(os.path.dirname(rundir), "latest_signals.csv"))  # MT5 bridge file
    return rundir


if __name__ == "__main__":
    cfg = dict(CONFIG)
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            cfg.update(json.load(f))
    out = run(cfg)
    rundir = save_run(out, cfg)
    s = out["stats"]
    print(f"asof {out['asof']}  Sharpe {s['sharpe']:.2f}  CAGR {s['cagr']:.1%}  MaxDD {s['max_dd']:.1%}")
    print(out["targets"].round(4).to_string())
    print(f"run saved -> {rundir}")
