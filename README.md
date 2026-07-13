# Slow Turtle Research

Systematic research program: from the book's 22/55-week SMA "Slow Turtle" to a
diversified, vol-targeted, FTMO-compatible trend-following engine.

**Premise investigated:** price momentum persists because institutional flows,
behavioral biases and slow information diffusion create trends that last longer
than expected. Every book parameter was treated as a hypothesis.

## How to run

```
pip install pandas numpy scipy matplotlib yfinance
python slow_turtle.py        # Phase 1+2 baseline + report
python phase3_ma_study.py    # ... any phase, each writes results/phaseN_report.md
python engine.py             # production run -> runs/<ts>/ + runs/latest_signals.csv
python test_slow_turtle.py && python test_engine.py   # sanity checks
```

Data: yfinance daily → weekly (W-FRI), cached in `data/`. 16 assets: 6 equity
indices, gold/silver/oil futures, 3 FX pairs, 4 sector ETFs (sectors instead of
single stocks to avoid survivorship bias). All backtests: signal at weekly close →
execution at next Monday open, open-to-open returns, long only.

## Findings ledger (each phase = one script + one report in results/)

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

## Production architecture

```
data (yfinance + cache)      slow_turtle.fetch_daily / to_weekly
  -> features / trend        common.ma (SMA/EMA/WMA/HMA/KAMA), atr
  -> signal                  fast MA > slow MA, lag 2 weeks
  -> sizing                  engine.sleeve_weights (EWMA vol target)
  -> portfolio               equal weight across sleeves
  -> risk overlays           phase11_ftmo.overlay (scale, dd-cut, vol brake)
  -> execution artifacts     runs/<ts>/{config,stats,equity,signals} + runs/latest_signals.csv
```

`engine.py` is fully config-driven (`python engine.py my_config.json`); every run is
reproducible and archived. The MT5 side consumes `runs/latest_signals.csv`
(name, ticker, target_weight, asof) — a minimal MQL5 bridge EA reads it weekly and
rebalances. Live-order routing deliberately left to the MT5 EA.
