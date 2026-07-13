"""Weekly cycle: refresh data, compute target weights, risk-check, execute
(dry-run by default), journal everything.

Usage:
    python run_weekly.py           # dry run
    python run_weekly.py --live    # requires MetaTrader5 package + terminal
"""
import argparse

from core import load_config, run, ftmo_check, intents_from, Journal
from mt5_connector import DryRunConnector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-refresh", action="store_true", help="use cached data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    journal = Journal()

    out = run(cfg, refresh=not args.no_refresh)

    # FTMO monitor from journaled equity marks (paper equity starts at 1.0)
    marks = [r for r in journal.read() if r["kind"] == "equity_mark"]
    eq_today = float(out["equity"].iloc[-1])
    eq_yesterday = marks[-1]["equity"] if marks else eq_today
    eq_peak = max([m["equity"] for m in marks] + [eq_today])
    risk = ftmo_check(eq_today, eq_yesterday, eq_peak, cfg)
    journal.log("equity_mark", {"equity": eq_today, "asof": out["asof"]})

    intents = intents_from(out, halt=risk["halt_new_entries"])
    journal.log("decision", {"asof": out["asof"], "risk": risk,
                             "targets": {i.asset: i.target_weight for i in intents}})

    if args.live:
        from mt5_connector import MT5Connector
        connector = MT5Connector(journal)
    else:
        connector = DryRunConnector(journal)
    connector.execute(intents)

    s = out["stats"]
    print(f"\nasof {out['asof']}  backtest: Sharpe {s['sharpe']:.2f}  "
          f"CAGR {s['cagr']:.1%}  MaxDD {s['max_dd']:.1%}")
    print(f"risk: daily {risk['daily_pnl']:+.2%}, dd {risk['total_dd']:+.2%}, "
          f"halt={risk['halt_new_entries']}")


if __name__ == "__main__":
    main()
