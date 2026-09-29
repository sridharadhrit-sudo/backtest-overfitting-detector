# Backtest Overfitting Detector

Backtest a thousand trading rules with no real edge, keep the best one, and it will
look excellent. You haven't found a strategy — you've found the right tail of a
sampling distribution. The best Sharpe ratio you find measures how many things you
tried, not how good the strategy is.

This repo builds the tool that quantifies that. Given a family of backtested
strategies, it reports the Deflated Sharpe Ratio, the Probability of Backtest
Overfitting, how many of the trials were genuinely independent, and how much more data
you would need before the result could be believed at all.

Based on Bailey & Lopez de Prado (2014), *The Deflated Sharpe Ratio*, and
Bailey, Borwein, Lopez de Prado & Zhu (2015), *The Probability of Backtest Overfitting*.

## The dashboard

`app.py` is an interactive front end to everything below. Pick a ticker, a history, the
grid of moving-average windows and a cost level, and every test reruns live: the winner
against the noise bar, the Deflated Sharpe Ratio under any assumed trial count, the
cross-validated overfitting probability with its degradation scatter, and — on demand —
the bootstrap null for exactly that family. A data tab gives every number in table form.

```bash
uv run streamlit run app.py
```

SPY from 2008 is bundled in `sample_data/`, so the default view works offline and loads
instantly; any other ticker downloads on first use.

## The case study

**The best of 815 moving-average variants on SPY since 2008 achieved an annualised
Sharpe of 0.41. Measured the usual way it looks significant — a probabilistic Sharpe
ratio of 95.6% against a zero benchmark. But when the same 815 variants are run on
resampled histories with every exploitable pattern destroyed, 82% of those histories
produce a better winner. The selection procedure that found it also fails
cross-validation 90% of the time, with the chosen variant's Sharpe collapsing from
+0.60 in sample to −0.02 out. It is, finally, worse than buying and holding the index.**

| | |
|---|---|
| Universe | SPY daily, Jan 2008 – Sep 2026 (4,411 days, 17.5 years) |
| Variants swept | 815 moving-average crossovers, long/short, 1bp costs |
| Winner | fast = 28, slow = 200 |
| Winner's annualised Sharpe | 0.4131 |
| PSR vs zero — *the usual claim* | 0.9559 |
| Noise threshold, formula at N = 815 | 0.3143 |
| Deflated Sharpe Ratio | 0.6584 |
| **Noise threshold, measured by bootstrap** | **0.5898** |
| **P(resampled history beats the winner)** | **0.82** |
| Probability of Backtest Overfitting | 0.898 |
| Winner in sample → out of sample | +0.60 → −0.02 |
| MinTRL vs the formula's threshold | 285 years |
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
p ≈ 1e-53. The diagnostic shows why — DSR divides by the standard error of a *single*
strategy's Sharpe (0.02822), but the *maximum* of many Sharpes varies far less than that
(0.01168), so the yardstick is 2.42× too wide and every z-score is shrunk accordingly.
DSR is therefore conservative: a value of 0.75 is *not* "75% likely real" but roughly a
1-in-20 result under pure noise, while a value above 0.95 is a strong signal precisely
because it almost never occurs by chance. Thresholds should be calibrated by simulation,
which is what `scripts/null_reference.py` does.

**Cross-validation convicts the procedure, not just the number.** Combinatorially
symmetric cross-validation over all 12,870 symmetric splits of the history gives a
Probability of Backtest Overfitting of **0.898** — the in-sample winner lands in the
bottom half out of sample nine times in ten. That is worse than the 0.5 a coin flip
would give, because taking the maximum of 815 noisy estimates preferentially selects
whichever variant received the kindest measurement error, and errors do not persist.
The degradation slope is **−0.56**: the better a variant looked in sample, the worse it
did out. PBO assumes no distributional model and never asks how many variants were
tried, so a flaw that breaks the DSR has no route to break it.

**Counting independent trials by structure would have flipped the verdict.** The 815
variants span only about 2.7 independent directions by a spectral measure, and cluster
into 3 groups at a 0.5 correlation threshold. Substituting N ≈ 3 into the deflation
raises the DSR from 0.66 to about 0.89 and makes the strategy look respectable. That
reasoning is wrong: those measures count independent directions in return space, not
independent opportunities for luck.

**Measuring the family's own null is the honest alternative — and the formula was too
lenient, not too harsh.** Resampling SPY's daily returns in blocks destroys trends and
reversals while preserving volatility clustering; rerunning the entire 815-variant sweep
on 300 such histories measures directly what "best of my variants, by luck alone" means,
with the variants exactly as correlated as they really are. The measured threshold is
**0.59** annualised against the formula's 0.31, and **82%** of resampled histories
produce a better winner than the real one. Two effects compound: the cross-sectional
spread of trial Sharpes is 46% wider on structureless data than on real SPY, whose long
bull trend pushes variants in a common direction; and even given that spread, the
observed maximum exceeds the Gaussian extreme-value prediction by about 18%, because the
cross-section of these Sharpes is not Gaussian. The result is stable at P = 0.70–0.82
across resampling block lengths from 1 to 63 days.

**One quantity in this repo is deliberately not quoted.** Inverting the measured maximum
to an "implied independent trial count" gives 22,142 with 21-day blocks and about 7,000
with an i.i.d. shuffle. Since E[max] grows like √(2 ln N), a 7% shift in the measured
maximum moves the implied N by a factor of three. The bootstrap P-value requires no such
inversion and is reported instead.

**Fat tails matter less than they look.** The winning variant has kurtosis 14.2 and
skewness −0.97, which sounds alarming, but together they inflate the Sharpe standard
error by only 1.3%. The skewness term contributes 11× more than the kurtosis term,
because kurtosis enters multiplied by the square of a small per-period Sharpe.

Every figure above comes from a seeded random number generator, so cloning this repo and
running the scripts reproduces them exactly — they are evidence rather than anecdote.

## Running it

```bash
uv run pytest -q                                # 44 tests, including the dashboard
uv run streamlit run app.py                     # the interactive dashboard
uv run python -m scripts.fig_null_world         # the noise threshold, simulated vs analytic
uv run python -m scripts.demo_sharpe            # what a Sharpe of 1.5 is actually worth
uv run python -m scripts.fig_dsr_calibration    # is the DSR well behaved under the null?
uv run python -m scripts.run_sweep              # 815 real backtests, and the verdict
uv run python -m scripts.null_reference         # that verdict against independent no-edge families
uv run python -m scripts.run_pbo                # cross-validated overfitting probability
uv run python -m scripts.fig_effective_trials   # how many trials were really independent
```

## Layout

- `src/units.py` — annualisation conversions, kept in one place
- `src/nulls.py` — simulating families with no edge; the expected-maximum threshold
- `src/sharpe.py` — Sharpe standard error, Probabilistic Sharpe Ratio, minimum track record length
- `src/dsr.py` — the Deflated Sharpe Ratio and family-level selection
- `src/data.py` — price download and caching; synthetic prices for offline tests
- `src/backtest.py` — the strategy family, costs, the grid sweep, and a 17× faster equivalent
- `src/pbo.py` — combinatorially symmetric cross-validation
- `src/effective_trials.py` — clustering, spectral and bootstrap measures of the trial count
- `src/charts.py` — chart specifications for the dashboard; presentation only
- `app.py` — the Streamlit dashboard; asks `src/` for every number and only draws
- `sample_data/` — bundled SPY prices so the dashboard runs offline
- `tests/` — 44 tests, including a look-ahead-bias check, analytic formulas verified against simulation, and a headless run of the dashboard
- `figures/` — generated plots

## Notes on method

Positions are shifted one day relative to the signal, so no trade uses a price that was
not yet observable; a test tampers with the final price and asserts no earlier position
moves. Transaction costs of 1bp are charged on turnover. The rule is long/short rather
than long/flat, so it cannot inherit the market's own drift and report it as skill. All
variants are trimmed to a common date range, so their Sharpes are comparable and the
cross-sectional dispersion means something. Tests never touch the network.

## Status

Stages 0–7 of 8 complete. Next: the written methods note.
