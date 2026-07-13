"""Sanity checks for the production engine core."""
import os
import tempfile
import numpy as np
from core import load_config, run, ftmo_check, RebalanceIntent, Journal
from mt5_connector import DryRunConnector


def test_engine_end_to_end():
    cfg = load_config()
    out = run(cfg)
    s = out["stats"]
    assert s["sharpe"] > 0.5, f"engine Sharpe degraded: {s['sharpe']:.2f}"
    assert s["max_dd"] > -0.10, f"engine maxDD outside FTMO limit: {s['max_dd']:.1%}"
    t = out["targets"]
    assert len(t) == len(cfg["universe"]["names"])
    assert (t >= 0).all() and t.sum() <= 1.01, "weights must be long-only, total <= 1"
    assert np.isfinite(out["equity"].iloc[-1])


def test_ftmo_check_halts():
    cfg = load_config()
    ok = ftmo_check(1.0, 1.0, 1.0, cfg)
    assert not ok["halt_new_entries"]
    daily_breach = ftmo_check(0.955, 1.0, 1.0, cfg)   # -4.5% day > 0.8*5% buffer
    assert daily_breach["halt_new_entries"]
    dd_breach = ftmo_check(0.91, 0.915, 1.0, cfg)     # -9% dd > 0.8*10% buffer
    assert dd_breach["halt_new_entries"]


def test_dry_run_journals_orders():
    path = os.path.join(tempfile.gettempdir(), "test_journal.jsonl")
    if os.path.exists(path):
        os.remove(path)
    journal = Journal(path)
    intents = [RebalanceIntent("2026-07-17", "SPY", "SPY", 0.0269, "weekly rebalance"),
               RebalanceIntent("2026-07-17", "GOLD", "GC=F", 0.0164, "weekly rebalance")]
    recs = DryRunConnector(journal).execute(intents)
    assert len(recs) == 2
    logged = journal.read()
    assert len(logged) == 2 and logged[0]["kind"] == "dry_run_order"
    assert logged[1]["intent"]["target_weight"] == 0.0164
    os.remove(path)


if __name__ == "__main__":
    test_ftmo_check_halts()
    test_engine_end_to_end()
    test_dry_run_journals_orders()
    print("engine sanity checks passed")
