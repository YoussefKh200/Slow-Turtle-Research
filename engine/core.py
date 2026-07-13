"""Production engine core: config -> weekly target weights.

Modular by function, not by file count: each stage is a pure function that can
be swapped without touching the others. State (equity marks, past decisions)
lives in the journal, not in memory.

    data -> sleeve_weight() -> portfolio -> overlay exposure -> RebalanceIntent

The engine never talks to a broker directly; it emits RebalanceIntent objects
that a connector (mt5_connector.MT5Connector or DryRunConnector) executes.
"""
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ENGINE_DIR, "..", "src"))

from data import UNIVERSE, load_universe          # noqa: E402
from backtest import ma, WEEKS_PER_YEAR           # noqa: E402
from stats import perf                            # noqa: E402
from phase11_ftmo import overlay                  # noqa: E402


def load_config(path=None):
    with open(path or os.path.join(ENGINE_DIR, "config.json")) as f:
        return json.load(f)


# ---------- sizing ----------
def sleeve_weight(weekly: pd.DataFrame, cfg) -> pd.Series:
    """Target weight series for one asset (signal x vol-target, before overlays)."""
    sg, sz = cfg["signal"], cfg["sizing"]
    c = weekly["Close"]
    sig = ma(c, sg["fast"], sg["ma_kind"]) > ma(c, sg["slow"], sg["ma_kind"])
    vol = c.pct_change().ewm(span=sz["ewma_span"], adjust=False).std() * np.sqrt(WEEKS_PER_YEAR)
    weight = (sz["sleeve_vol_target"] / vol).clip(upper=sz["sleeve_cap"])
    return (sig * weight).fillna(0.0)


# ---------- portfolio ----------
def run(cfg, refresh: bool = False) -> dict:
    """Full-history backtest + current target weights under the FTMO overlays."""
    ro = cfg["risk_overlays"]
    uni = {a: w for a, w in load_universe(refresh=refresh).items()
           if a in cfg["universe"]["names"]}
    n = len(uni)
    weights = pd.concat({a: sleeve_weight(w, cfg) for a, w in uni.items()},
                        axis=1, sort=True).ffill().fillna(0.0) / n
    rets = pd.concat({a: w["Open"].pct_change() for a, w in uni.items()},
                     axis=1, sort=True).dropna()
    lagged = weights.shift(cfg["signal"]["lag_weeks"]).reindex(rets.index)
    warmup = cfg["signal"]["slow"] * 2
    port = (rets * lagged).sum(axis=1).iloc[warmup:]
    strat = overlay(port, cfg["portfolio"]["portfolio_scale"], ro["dd_scaling"],
                    ro["vol_brake"], target=cfg["sizing"]["sleeve_vol_target"])
    expo = strat.iloc[-1] / port.iloc[-1] if port.iloc[-1] != 0 \
        else cfg["portfolio"]["portfolio_scale"]
    return {"stats": perf(strat), "equity": (1 + strat).cumprod(),
            "targets": (weights.iloc[-1] * expo).rename("target_weight"),
            "asof": str(weights.index[-1].date())}


# ---------- risk ----------
def ftmo_check(eq_today: float, eq_yesterday: float, eq_peak: float, cfg) -> dict:
    f = cfg["ftmo"]
    daily = eq_today / eq_yesterday - 1
    dd = eq_today / eq_peak - 1
    halt = (daily < -f["max_daily_loss"] * f["halt_buffer"]
            or dd < -f["max_total_drawdown"] * f["halt_buffer"])
    return {"daily_pnl": daily, "total_dd": dd, "halt_new_entries": halt}


# ---------- intents & journal ----------
@dataclass
class RebalanceIntent:
    asof: str
    asset: str
    ticker: str
    target_weight: float
    reason: str


def intents_from(out: dict, halt: bool) -> list[RebalanceIntent]:
    reason = "ftmo_halt: targets forced to 0" if halt else "weekly rebalance"
    return [RebalanceIntent(out["asof"], a, UNIVERSE[a],
                            0.0 if halt else round(float(tw), 4), reason)
            for a, tw in out["targets"].items()]


class Journal:
    """Append-only jsonl journal; the engine's only state."""

    def __init__(self, path=None):
        self.path = path or os.path.join(ENGINE_DIR, "journal.jsonl")

    def log(self, kind: str, payload: dict) -> dict:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def read(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
