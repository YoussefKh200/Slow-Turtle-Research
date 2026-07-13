"""Daily cycle for the blended TF+MR book (phase13b/13c 80/20).

Runs DAILY (not weekly): MR counts its max-hold in bars, so the combined
engine must tick once per trading day for MR's day-count to be right. TF's
weekly targets simply re-emit unchanged until they roll on Friday.

Dry-run only for now -- live routing waits on a broker account, same as
run_weekly.py. Uses a separate journal so it never mixes with the pure-TF
engine's state.

Usage:
    python run_combined.py
    python run_combined.py --no-refresh    # use cached data
"""
import argparse
import os

from core import load_config, ftmo_check, Journal
from combined import combined_targets
from mt5_connector import DryRunConnector

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-refresh", action="store_true", help="use cached data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    journal = Journal(os.path.join(ENGINE_DIR, "journal_combined.jsonl"))

    out = combined_targets(cfg, journal, refresh=not args.no_refresh)

    # FTMO monitor on the combined paper equity. Real equity comes from fills,
    # which don't exist until live; in dry-run it stays 1.0, so the halt logic
    # is wired and journalled but never actually trips here.
    # ponytail: mark from live fills once a broker account exists.
    marks = [r for r in journal.read() if r["kind"] == "equity_mark"]
    eq_today = 1.0
    eq_yesterday = marks[-1]["equity"] if marks else eq_today
    eq_peak = max([m["equity"] for m in marks] + [eq_today])
    risk = ftmo_check(eq_today, eq_yesterday, eq_peak, cfg)

    journal.log("equity_mark", {"equity": eq_today, "asof": out["asof"]})
    for rec in out["mr_journal_records"]:
        journal.log(rec["kind"], rec)
    journal.log("decision", {"asof": out["asof"], "risk": risk,
                             "gross_exposure": out["gross_exposure"],
                             "targets": {i.asset: i.target_weight for i in out["intents"]}})

    DryRunConnector(journal).execute(out["intents"])

    s = out["stats"]
    print(f"\nasof {out['asof']}  TF-book backtest: Sharpe {s['sharpe']:.2f}  "
          f"CAGR {s['cagr']:.1%}  MaxDD {s['max_dd']:.1%}")
    print(f"combined gross exposure {out['gross_exposure']:.1%}  "
          f"halt={risk['halt_new_entries']}")


if __name__ == "__main__":
    main()
