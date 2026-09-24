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

## Findings so far

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
by simulation rather than borrowed from conventions that assume uniformity.

Every figure above is produced by a seeded random number generator, so cloning this repo
and running the scripts reproduces them exactly — they are evidence rather than anecdote.

## Running it

```bash
uv run pytest -q                              # 19 tests
uv run python -m scripts.fig_null_world       # the noise threshold, simulated vs analytic
uv run python -m scripts.demo_sharpe          # what a Sharpe of 1.5 is actually worth
uv run python -m scripts.fig_dsr_calibration  # the calibration study
```

## Layout

- `src/units.py` — annualisation conversions, kept in one place
- `src/nulls.py` — simulating families with no edge; the expected-maximum threshold
- `src/sharpe.py` — Sharpe standard error, Probabilistic Sharpe Ratio, minimum track record length
- `src/dsr.py` — the Deflated Sharpe Ratio and family-level selection
- `tests/` — including checks of the analytic formulas against brute-force simulation
- `figures/` — generated plots

## Status

In progress. Stages 0–3 of 8 complete: project skeleton, null-world simulation,
Sharpe estimation machinery, and the Deflated Sharpe Ratio with its calibration study.

Next: a real strategy family — sweep a parameterised rule over real price data, then
put the winner through the detector.
