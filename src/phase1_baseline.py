"""Phase 1+2: Slow Turtle baseline (22/55-week SMA cross, long only, Monday-open
execution) and the full performance analysis.

Run: python src/phase1_baseline.py -> results/phase1/PHASE1_REPORT.md + csv artifacts
"""
import os
import pandas as pd
from data import UNIVERSE, load_universe
from backtest import cross_backtest
from stats import full_metrics, profit_concentration

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase1")


def run_all() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, all_trades, extras = [], [], {}
    for name, w in load_universe().items():
        strat_ret, trades, pos = cross_backtest(w)
        m = full_metrics(strat_ret, trades, pos)
        if not m:
            print(f"  SKIP {name}: insufficient history")
            continue
        extras[name] = {"yearly": m.pop("yearly"), "equity": m.pop("equity"),
                        "concentration": profit_concentration(m.pop("trades"))}
        rows.append({"asset": name, "ticker": UNIVERSE[name], **m})
        all_trades.append(trades.assign(asset=name))
        print(f"  {name}: CAGR {m['cagr']:.1%}, Sharpe {m['sharpe']:.2f}, "
              f"MaxDD {m['max_dd']:.1%}, {m['n_trades']} trades")
    return (pd.DataFrame(rows).set_index("asset"),
            pd.concat(all_trades, ignore_index=True), extras)


def write_report(df: pd.DataFrame, trades: pd.DataFrame, extras: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "metrics.csv"))
    trades.to_csv(os.path.join(OUT, "trades.csv"), index=False)

    pct = lambda x: f"{x:.1%}" if pd.notna(x) else "-"
    num = lambda x: f"{x:.2f}" if pd.notna(x) else "-"
    lines = [
        "# Phase 1 — Slow Turtle Baseline\n",
        "System: 22/55-week SMA cross, weekly candles, long only, next-Monday-open execution. No costs.\n",
        "## Per-asset performance\n",
        "| Asset | Period | CAGR | Vol | Sharpe | Sortino | Calmar | MaxDD | PF | WinRate | Trades | AvgHold(w) | Exposure | Neg.Yrs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a, r in df.iterrows():
        lines.append(
            f"| {a} | {r['start']}->{r['end']} | {pct(r['cagr'])} | {pct(r['ann_vol'])} | {num(r['sharpe'])} "
            f"| {num(r['sortino'])} | {num(r['calmar'])} | {pct(r['max_dd'])} | {num(r['profit_factor'])} "
            f"| {pct(r['win_rate'])} | {int(r['n_trades'])} | {num(r['avg_weeks_held'])} | {pct(r['exposure'])} "
            f"| {int(r['negative_years'])}/{int(r['total_years_counted'])} |")

    lines += ["\n## Trade P&L structure\n",
              "| Asset | AvgWin | AvgLoss | LargestWin | LargestLoss | Top10% trades = %profit | Trades for 100% profit | Underwater max (yrs) |",
              "|---|---|---|---|---|---|---|---|"]
    for a, r in df.iterrows():
        c = extras[a]["concentration"]
        share = pct(c.get("top10pct_share")) if "top10pct_share" in c else c.get("note", "-")
        nfull = f"{c['n_trades_for_100pct']}/{c['n_trades']}" if "n_trades_for_100pct" in c else "-"
        lines.append(f"| {a} | {pct(r['avg_winner'])} | {pct(r['avg_loser'])} | {pct(r['largest_winner'])} "
                     f"| {pct(r['largest_loser'])} | {share} | {nfull} | {num(r['longest_underwater_yrs'])} |")

    lines += ["\n## Yearly returns (strategy)\n"]
    yearly = pd.DataFrame({a: e["yearly"] for a, e in extras.items()})
    lines.append(yearly.map(lambda x: f"{x:.1%}" if pd.notna(x) else "").to_markdown())

    with open(os.path.join(OUT, "PHASE1_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport -> {os.path.join(OUT, 'PHASE1_REPORT.md')}")


if __name__ == "__main__":
    df, trades, extras = run_all()
    write_report(df, trades, extras)
