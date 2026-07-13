"""Sanity checks for the blended TF+MR engine (combined.py).

Uses cached data only (refresh=False) so it never hits the network.
"""
import os
import tempfile

from core import load_config, Journal
import combined


def _temp_journal():
    path = os.path.join(tempfile.gettempdir(), "test_combined_journal.jsonl")
    if os.path.exists(path):
        os.remove(path)
    return Journal(path), path


def test_mr_position_replay():
    """enter_long -> Position; hold -> bars_held increments; exit -> None."""
    _, mr_core, _ = combined.load_mr(load_config()["blend"]["mr_repo"])
    recs = [
        {"kind": "mr_decision", "ticker": "QQQ",
         "intent": {"action": "enter_long", "timestamp": "2026-01-05",
                    "ref_price": 100.0, "fraction": 0.5}},
        {"kind": "mr_decision", "ticker": "QQQ",
         "intent": {"action": "hold", "timestamp": "2026-01-06"}},
        {"kind": "mr_decision", "ticker": "SPY",  # different ticker, must be ignored
         "intent": {"action": "enter_long", "timestamp": "2026-01-06",
                    "ref_price": 50.0, "fraction": 0.3}},
    ]
    pos = combined.mr_position(recs, "QQQ", mr_core)
    assert pos is not None and pos.entry_price == 100.0 and pos.fraction == 0.5
    assert pos.bars_held == 1, "one hold after entry -> bars_held == 1"

    closed = combined.mr_position(recs + [
        {"kind": "mr_decision", "ticker": "QQQ",
         "intent": {"action": "exit", "timestamp": "2026-01-07"}}], "QQQ", mr_core)
    assert closed is None, "exit clears the position"


def test_tf_scaled_by_blend_weight():
    """Each TF sleeve's combined weight == its standalone weight x tf_weight."""
    import core as tf
    cfg = load_config()
    journal, path = _temp_journal()
    try:
        standalone = tf.run(cfg, refresh=False)["targets"]
        out = combined.combined_targets(cfg, journal, refresh=False)
        tf_intents = {i.asset: i.target_weight for i in out["intents"]
                      if not i.asset.startswith("MR_")}
        for asset, w in standalone.items():
            expected = round(float(w * cfg["blend"]["tf_weight"]), 4)
            assert abs(tf_intents[asset] - expected) < 1e-9, \
                f"{asset}: {tf_intents[asset]} != {expected}"
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_mr_held_weight_formula():
    """A held MR position emits weight == fraction x (1-tf_weight)/n_mr x mr_vol_match."""
    cfg = load_config()
    b = cfg["blend"]
    journal, path = _temp_journal()
    try:
        # seed an open QQQ position with entry far ABOVE any real price, so
        # decide() returns "hold" (close < entry) rather than "exit".
        frac = 0.5
        journal.log("mr_decision", {"kind": "mr_decision", "ticker": "QQQ",
                    "intent": {"action": "enter_long", "timestamp": "2020-01-01",
                               "ref_price": 1e9, "fraction": frac}})
        out = combined.combined_targets(cfg, journal, refresh=False)
        mr_qqq = next(i for i in out["intents"] if i.asset == "MR_QQQ")
        expected = round(frac * (1 - b["tf_weight"]) / len(b["mr_tickers"]) * b["mr_vol_match"], 4)
        assert "hold" in mr_qqq.reason, f"expected a held position, got: {mr_qqq.reason}"
        assert abs(mr_qqq.target_weight - expected) < 1e-9, \
            f"MR_QQQ weight {mr_qqq.target_weight} != {expected}"
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    test_mr_position_replay()
    test_tf_scaled_by_blend_weight()
    test_mr_held_weight_formula()
    print("combined engine sanity checks passed")
