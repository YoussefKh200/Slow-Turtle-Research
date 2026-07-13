# Phase 7 — Regime Filters

Filter ANDed with baseline 22/55 signal. Whipsaw = round trip held < 8 weeks.
16 assets, common warmup, no costs.

|                             |   median_sharpe |   median_maxdd |   total_trades |   total_whipsaws |   avg_exposure |
|:----------------------------|----------------:|---------------:|---------------:|-----------------:|---------------:|
| MA dist>1%                  |            0.53 |          -0.34 |            220 |               15 |           0.62 |
| ADX>20                      |            0.52 |          -0.33 |            560 |              131 |           0.54 |
| baseline (none)             |            0.52 |          -0.36 |            198 |               11 |           0.69 |
| strength pctile>30%         |            0.49 |          -0.33 |            258 |               19 |           0.57 |
| ADX>25                      |            0.45 |          -0.31 |            567 |              130 |           0.44 |
| vol calm (ATR14<ATR52)      |            0.44 |          -0.28 |            625 |              285 |           0.35 |
| MA dist>3%                  |            0.42 |          -0.34 |            241 |               16 |           0.49 |
| vol expanding (ATR14>ATR52) |            0.36 |          -0.31 |            584 |              267 |           0.34 |

