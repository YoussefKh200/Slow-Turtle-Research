"""Sanity checks for the production engine."""
import numpy as np
from engine import CONFIG, run


def test_engine_end_to_end():
    out = run(dict(CONFIG))
    s = out["stats"]
    assert s["sharpe"] > 0.5, f"engine Sharpe degraded: {s['sharpe']:.2f}"
    assert s["max_dd"] > -0.10, f"engine maxDD outside FTMO limit: {s['max_dd']:.1%}"
    t = out["targets"]
    assert len(t) == len(CONFIG["universe"])
    assert (t >= 0).all() and t.sum() <= 1.01, "weights must be long-only, total <= 1"
    assert np.isfinite(out["equity"].iloc[-1])


if __name__ == "__main__":
    test_engine_end_to_end()
    print("engine sanity checks passed")
