"""Blended two-book engine: TF trend book + MR mean-reversion book at the
Phase 13b/13c validated 80/20 split.

The blend is a capital allocation between two independently-researched,
near-uncorrelated books (corr -0.06), NOT a merge of signals:

    combined = 0.8 * TF_ftmo  +  0.2 * MR_volmatched      (phase13b)

carried to live weights as, per config["blend"]:
    TF sleeve i : w_tf_i * tf_weight
    MR ticker j : f_mr_j * (1 - tf_weight)/n_mr * mr_vol_match

where w_tf_i is TF's live FTMO-scaled target weight, f_mr_j is MR's live
vol-targeted fraction, and mr_vol_match = sigma_tf/sigma_mr equalises the two
books' realised vol before the split (MR is ~3x more volatile; see config).

Design notes:
- The MR program's engine is reused as-is via importlib under isolated module
  names (mr_core / mr_data), so this repo's own `core`/`data` modules don't
  collide with MR's same-named ones. No fork, no reimplementation of any
  signal -- if MR's research changes, its engine code changes here too.
- MR state (open position per ticker) is replayed from THIS engine's combined
  journal, not MR's -- the combined book owns its own state.
- Run this DAILY (see run_combined.py): MR's max_hold is counted in bars, so a
  weekly cadence would stretch a 20-day hold into 20 weeks. TF's weekly targets
  are stable intra-week, so daily runs re-emit them unchanged until they roll.
"""
import importlib.util
import os

import core as tf  # this engine's TF core
from core import RebalanceIntent

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mr(mr_repo: str):
    """Isolated-load the MR program's engine core + data loader + config."""
    root = os.path.join(ENGINE_DIR, mr_repo)
    mr_data = _load("mr_data", os.path.join(root, "src", "data.py"))
    mr_core = _load("mr_core", os.path.join(root, "engine", "core.py"))
    mr_cfg = mr_core.load_config(os.path.join(root, "engine", "config.json"))
    return mr_data, mr_core, mr_cfg


def mr_position(records: list[dict], ticker: str, mr_core):
    """Reconstruct one MR ticker's open position by replaying combined-journal
    mr_decision records for that ticker (append-only state)."""
    pos = None
    for rec in records:
        if rec.get("kind") != "mr_decision" or rec.get("ticker") != ticker:
            continue
        it = rec["intent"]
        if it["action"] == "enter_long":
            pos = mr_core.Position(it["timestamp"], it["ref_price"], it["fraction"], 0)
        elif it["action"] == "exit":
            pos = None
        elif it["action"] == "hold" and pos is not None:
            pos.bars_held += 1
    return pos


def combined_targets(cfg, journal, refresh: bool = True) -> dict:
    """Today's blended book. Returns TF stats, the merged RebalanceIntent list,
    and the raw MR intents to journal (so per-ticker state replays next run)."""
    b = cfg["blend"]
    records = journal.read()

    # --- TF book: reuse the existing engine unchanged, then scale ---
    tf_out = tf.run(cfg, refresh=refresh)
    intents = [RebalanceIntent(tf_out["asof"], asset, tf.UNIVERSE[asset],
                               round(float(w * b["tf_weight"]), 4), "blend: TF 80%")
               for asset, w in tf_out["targets"].items()]

    # --- MR book: reuse MR's decide() per ticker, scale by the vol-matched split ---
    mr_data, mr_core, mr_cfg = load_mr(b["mr_repo"])
    mr_mult = (1 - b["tf_weight"]) / len(b["mr_tickers"]) * b["mr_vol_match"]
    mr_journal_records = []
    for tk in b["mr_tickers"]:
        df = mr_data.get_data(tk, refresh=refresh)
        pos = mr_position(records, tk, mr_core)
        cfg_tk = {**mr_cfg, "instrument": {**mr_cfg["instrument"], "ticker": tk}}
        it = mr_core.decide(df, pos, cfg_tk)
        held = it.action in ("enter_long", "hold")
        weight = round(float(it.fraction * mr_mult), 4) if held else 0.0
        intents.append(RebalanceIntent(it.timestamp, f"MR_{tk}", tk, weight,
                                       f"blend: MR 20% ({it.action})"))
        mr_journal_records.append({"kind": "mr_decision", "ticker": tk,
                                   "intent": it.__dict__})

    return {"stats": tf_out["stats"], "asof": tf_out["asof"], "intents": intents,
            "mr_journal_records": mr_journal_records,
            "gross_exposure": round(sum(abs(i.target_weight) for i in intents), 4)}


if __name__ == "__main__":
    raise SystemExit("combined.py is a library -- run:  python run_combined.py [--mt5] [--live]")
