# Backtest Overfitting Detector

Backtest a thousand trading rules with no real edge, keep the best one, and it will
look excellent. You haven't found a strategy — you've found the right tail of a
sampling distribution. The best Sharpe ratio you find measures how many things you
tried, not how good the strategy is.

This repo builds the tool that quantifies that. Given a family of backtested
strategies, it reports the Deflated Sharpe Ratio, the Probability of Backtest
Overfitting, and how much more data you would need before the result could be
believed at all.

Based on Bailey & Lopez de Prado (2014), *The Deflated Sharpe Ratio*, and
Bailey, Borwein, Lopez de Prado & Zhu (2015), *The Probability of Backtest Overfitting*.

## The case study

**The best of 815 moving-average variants on SPY since 2008 achieved an annualised
Sharpe of 0.41. Measured the usual way it looks significant — a probabilistic Sharpe
ratio of 95.6% against a zero benchmark. But 815 worthless strategies reach 0.31 by
luck alone, and against a null distribution of the same size and length, 15% of no-edge
families do at least as well. Establishing the remaining margin would take 285 years of
data. It is also worse than buying and holding the index.**

The full numbers, from `scripts/run_sweep.py` and `scripts/null_reference.py`:

| | |
|---|---|
| Universe | SPY daily, Jan 2008 – Sep 2026 (4,411 days, 17.5 years) |
| Variants swept | 815 moving-average crossovers, long/short, 1bp costs |
| Winner | fast = 28, slow = 200 |
| Winner's annualised Sharpe | 0.4131 |
| PSR vs zero — *the usual claim* | 0.9559 |
| Noise threshold for 815 trials | 0.3143 |
| **Deflated Sharpe Ratio** | **0.6584** |
| Percentile in a matched null | 85.3rd (p = 0.147) |
| MinTRL vs the threshold | 285 years |
| Median variant | 0.2092 |
| Buy and hold SPY | 0.6432 |

The winner is long/short and so carries little market exposure, which the bare
comparison against buy-and-hold does not reflect. As a claim to have found something,
though, it is finished.

## Findings

**A thousand strategies with exactly zero edge, over five years of daily data, produce
a best annualised Sharpe of 1.46.** Simulation gives 1.4600; the closed-form
extreme-value expression gives 1.4557. Most published backtests would not clear that bar.

**Deflation works, and undeflated confidence is worthless.** Run on 1,000 families in
which nothing has any edge, the Probabilistic Sharpe Ratio declared the winner
significant at the 95% level in **99.4%** of runs. Deflated against the noise threshold,
that false-positive rate fell to **0 out of 1,000**.

**But the Deflated Sharpe Ratio is not a p-value.** Under the null it is centred
correctly (mean 0.4950) yet far too narrow: its 5th and 95th percentiles are 0.29 and
0.75, against 0.05 and 0.95 for a uniform statistic, and a KS test rejects uniformity at
p ≈ 1e-53. The cause is visible in the diagnostic output — DSR divides by the standard
error of a *single* strategy's Sharpe (0.02822), but the *maximum* of many Sharpes varies
far less than that (0.01168), so the yardstick is 2.42× too wide and every z-score is
shrunk accordingly.

The practical consequences: DSR is conservative and errs toward silence, which is the
right direction for a scepticism tool; a DSR of 0.75 is *not* "75% likely real" but
roughly a 1-in-20 result under pure noise; and a DSR above 0.95 is a strong signal
precisely because it essentially never occurs by chance. Thresholds should be calibrated
by simulation rather than borrowed from conventions that assume uniformity — which is
what `scripts/null_reference.py` does for the case study above.

**Fat tails matter less than they look.** The winning variant has kurtosis 14.2 and
skewness −0.97, which sounds alarming, but together they inflate the Sharpe standard
error by only 1.3%. The skewness term contributes 11× more than the kurtosis term,
because kurtosis enters multiplied by the square of a small per-period Sharpe.

Every figure above comes from a seeded random number generator, so cloning this repo and
running the scripts reproduces them exactly — they are evidence rather than anecdote.

## Running it

```bash
uv run pytest -q                              # 28 tests
uv run python -m scripts.fig_null_world       # the noise threshold, simulated vs analytic
uv run python -m scripts.demo_sharpe          # what a Sharpe of 1.5 is actually worth
uv run python -m scripts.fig_dsr_calibration  # is the DSR well behaved under the null?
uv run python -m scripts.run_sweep            # 815 real backtests, and the verdict
uv run python -m scripts.null_reference       # where that verdict sits among no-edge families
```

## Layout

- `src/units.py` — annualisation conversions, kept in one place
- `src/nulls.py` — simulating families with no edge; the expected-maximum threshold
- `src/sharpe.py` — Sharpe standard error, Probabilistic Sharpe Ratio, minimum track record length
- `src/dsr.py` — the Deflated Sharpe Ratio and family-level selection
- `src/data.py` — price download and caching; synthetic prices for offline tests
- `src/backtest.py` — the strategy family, costs, and the grid sweep
- `tests/` — including a look-ahead-bias test and checks of analytic formulas against simulation
- `figures/` — generated plots

## Status

In progress. Stages 0–4 of 8 complete.

Next: the Probability of Backtest Overfitting via combinatorially symmetric
cross-validation — a second, non-parametric line of attack on the same question — then
an honest count of how many of those 815 variants were genuinely independent.
