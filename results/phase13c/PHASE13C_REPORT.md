# Phase 13c — Blend robustness (TF 80% / MR 20%)

## Out-of-sample stability (disjoint 5y windows)

|           |   sharpe |   cagr |   max_dd |
|:----------|---------:|-------:|---------:|
| 2006-2011 |     0.72 |   0.02 |    -0.03 |
| 2011-2016 |     0.93 |   0.02 |    -0.02 |
| 2016-2021 |     1.49 |   0.03 |    -0.03 |
| 2021-2025 |     1.12 |   0.02 |    -0.03 |

## Monte Carlo (26-week block bootstrap, 2000 paths)

- Sharpe: 5th pct 0.69, median 1.05, 95th 1.43
- MaxDD: median -3.9%, 95th pct worst -5.8%
- P(maxDD worse than -10%): 0.1%

## Deflated Sharpe Ratio

- Observed Sharpe 1.07 over 1068 weeks, deflated for 514 trials (TF's 500 + 7 phase13 blends + 7 phase13b blends):
- **DSR = 0.956**

## TF-side execution stress carried through the blend

|                               |   sharpe |   cagr |   max_dd |
|:------------------------------|---------:|-------:|---------:|
| clean                         |     1.07 |   0.02 |    -0.03 |
| +1w execution delay           |     1.11 |   0.02 |    -0.03 |
| 10bp per turn                 |     1.05 |   0.02 |    -0.03 |
| 25bp per turn (spread stress) |     1.01 |   0.02 |    -0.03 |

Verdict: PASS if 5y windows stay positive, DSR > 0.9, stress Sharpe stays clear of the
pure-TF baseline (0.94).
