# Phase 12 — Robustness Testing

## Out-of-sample stability (disjoint 5y windows, no fitting anywhere)

|           |   sharpe |   cagr |   max_dd |
|:----------|---------:|-------:|---------:|
| 2006-2011 |     0.79 |   0.02 |    -0.03 |
| 2011-2016 |     0.78 |   0.02 |    -0.03 |
| 2016-2021 |     1.18 |   0.03 |    -0.04 |
| 2021-2025 |     0.93 |   0.02 |    -0.03 |

## Monte Carlo (26-week block bootstrap, 2000 paths)

- Sharpe: 5th pct 0.58, median 0.91, 95th 1.26
- MaxDD: median -4.6%, 95th pct worst -6.9%
- P(maxDD worse than -10%): 0.2%

## Execution stress

|                               |   sharpe |   cagr |   max_dd |
|:------------------------------|---------:|-------:|---------:|
| clean                         |     0.94 |   0.02 |    -0.04 |
| +1w execution delay           |     0.98 |   0.03 |    -0.04 |
| 10bp per turn                 |     0.91 |   0.02 |    -0.04 |
| 25bp per turn (spread stress) |     0.88 |   0.02 |    -0.04 |
| delay + 25bp                  |     0.92 |   0.02 |    -0.04 |

## Deflated Sharpe Ratio

- Observed Sharpe 0.94 over 1068 weeks, deflated for 500 trials:
- **DSR = 0.875** (probability the true Sharpe > 0 after selection bias)

## Parameter perturbation (+-20%)

|                            |   sharpe |   cagr |   max_dd |
|:---------------------------|---------:|-------:|---------:|
| fast=22 slow=55 target=10% |     0.94 |   0.02 |    -0.04 |
| fast=18 slow=44 target=10% |     0.88 |   0.02 |    -0.04 |
| fast=26 slow=66 target=10% |     0.92 |   0.02 |    -0.04 |
| fast=18 slow=66 target=10% |     0.92 |   0.02 |    -0.04 |
| fast=26 slow=44 target=10% |     0.87 |   0.02 |    -0.04 |
| fast=22 slow=55 target=8%  |     0.89 |   0.02 |    -0.03 |
| fast=22 slow=55 target=12% |     0.93 |   0.03 |    -0.04 |

Verdict: PASS if 5y windows stay positive, DSR > 0.9, stress Sharpe > 0.5,
and perturbed Sharpe stays within ~0.15 of base.
