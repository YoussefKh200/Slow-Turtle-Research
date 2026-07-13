# Slow Turtle Research

Systematic research program: from the book's 22/55-week SMA "Slow Turtle" to a
diversified, vol-targeted, FTMO-compatible trend-following engine.

**Premise investigated:** price momentum persists because institutional flows,
behavioral biases and slow information diffusion create trends that last longer
than expected. Every book parameter was treated as a hypothesis.

## Layout

```
data/      committed daily OHLC CSVs (yfinance cache, raw ticker filenames)
src/       shared modules (data.py, backtest.py, stats.py) + one script per phase
results/   results/phaseN/PHASEN_REPORT.md + charts/ + csv artifacts per phase
engine/    production engine: config.json, core.py, mt5_connector.py, run_weekly.py
```

## How to run

```
pip install pandas numpy scipy matplotlib yfinance
python src/phase1_baseline.py     # any phase -> results/phaseN/PHASEN_REPORT.md
python src/test_backtest.py       # backtest sanity checks
python engine/run_weekly.py       # weekly cycle, dry-run (--live for MT5)
python engine/test_engine.py && python engine/test_mt5_order.py
```

Data: yfinance daily → weekly (W-FRI), cached in `data/`. 16 assets: 6 equity
indices, gold/silver/oil futures, 3 FX pairs, 4 sector ETFs (sectors instead of
single stocks to avoid survivorship bias). All backtests: signal at weekly close →
execution at next Monday open, open-to-open returns, long only.

## Findings ledger (each phase = one script in src/ + one report in results/)

| Phase | Question | Verdict |
|---|---|---|
| 1-2 | Book baseline | Works on equities/gold (Sharpe 0.4-0.8) but maxDD 30-47%, profits concentrated in few huge trades, **fails on long-only FX**, underwater up to 21 years |
| 3 | Are 22/55 special? | No — the whole fast/slow grid is a flat plateau (93% of combos Sharpe>0.3). SMA=EMA=WMA; HMA worse. 22/55 already sits at the 90th percentile. **Keep it** |
| 4 | Better trend signal? | No family dominates: MA cross = 12-1 momentum = regression slope (~0.52 median Sharpe). Donchian slightly worse. Trend-*strength* gates (R², ER) hurt standalone |
| 5 | Vol-adjusted sizing | Sharpe unchanged, but **maxDD halves** at 10% EWMA vol target. Adopted: EWMA-26, 10% target, cap 1x |
| 6 | Diversification | Avg pairwise strategy corr 0.36; 16-sleeve portfolio: vol 6%, maxDD -12% vs -24% concentrated. **Biggest single improvement in the program** |
| 7 | Regime filters | **Rejected** — ADX/vol gates add 10x whipsaws, no Sharpe gain. The slow cross already is a regime filter |
| 8 | Entry timing | Irrelevant on weekly (all variants 0.51-0.53). Keep Monday open |
| 9 | Exits | Wide 4x ATR trail mildly improves Calmar (0.19 vs 0.15). Tight stops destroy trend capture (Phase 10 explains why) |
| 10 | MAE/MFE | 27% of winners first go >5% underwater; stops tighter than ~10% MAE kill winners. Risk belongs at **portfolio** level. Expect up to 6 consecutive losers per asset |
| 11 | FTMO version | 0.6x scale + linear dd-cut (flat at -8%) + vol brake: **maxDD -3.6%, Sharpe 0.94**, zero weeks < -2%. Room to lever ~2x inside a 10% limit |
| 12 | Robustness | All disjoint 5y windows positive (0.78-1.18); insensitive to 25bp costs & 1w delay; parameter perturbation ±20% stays within 0.07 Sharpe; MC bootstrap fine. **DSR 0.875** — below the 0.9 ideal, flagged honestly |
| 13 | + Mean reversion | TF/MR corr 0.25 (good) but the simple MR sleeve is too weak (Sharpe 0.22) — blending dilutes. Needs the dedicated MR program's sleeve before allocation |

## Production engine

```
data -> sleeve_weight() -> portfolio -> overlay exposure -> RebalanceIntent -> connector
```

- `engine/config.json` — every parameter cites the phase that selected it.
- `engine/core.py` — pure functions; state lives in the append-only `journal.jsonl`.
- `engine/run_weekly.py` — weekly cycle: refresh data, targets, FTMO check, execute.
  Dry-run by default; `--live` uses the MetaTrader5 package (`mt5_connector.MT5Connector`,
  FTMO CFD symbol map). Schedule `run_weekly.bat` for Monday pre-open.
