# Methods note: how much of a backtest is luck?

A companion to the [README](README.md) and the [live dashboard](https://backtest-overfitting-detector.streamlit.app/).
The README states what was found; this note explains how, what went wrong along the
way, and what the results do and do not show.

## The ninety-second version

If you backtest enough trading rules on the same data, one of them will look good by
luck alone — and the more rules you try, the better the luckiest one looks. So a high
Sharpe ratio on its own tells you as much about how hard you searched as about how good
the strategy is.

I built a tool that measures that. I tested it first on simulated strategies with no
edge at all, then pointed it at a real case: 815 variants of a moving-average rule on
the S&P 500 since 2008. The best one scored a Sharpe of 0.41, and measured the usual way
it looks significant at 95.6%. But when I rerun the same 815 strategies on histories with
every exploitable pattern destroyed, 82% of those histories produce a better winner. The
procedure that picked it also fails cross-validation nine times in ten: its Sharpe falls
from 0.60 on the data that chose it to −0.02 on data that didn't. And it loses to simply
holding the index.

The most useful thing I learned was methodological. My first estimate of the noise bar
came from a published formula that assumes the 815 tries were independent. I predicted
that correcting for their correlation would make the bar *easier* to clear. I measured it
instead, and it was the other way round — the formula had been too lenient. The fix for
an unconvincing backtest is almost never more data. It is searching less.

---

## 1. The question

A backtest runs a trading rule over historical prices as if it had been traded at the
time, and records the returns it would have made. The usual summary is the **Sharpe
ratio**: average return divided by its volatility, a signal-to-noise ratio for a return
stream. By convention it is quoted per year.

The problem is that nobody backtests one rule. They try many — different parameters,
different signals — and report the best. Each backtest's Sharpe ratio is a noisy
*estimate*, so picking the largest of many estimates picks whichever one received the
most favourable noise. The winner's number is inflated by construction, even if every
rule is worthless.

This project asks: **given a family of backtested strategies and the one that won, how
much of the winner's performance survives once you account for having searched?**

It answers with three independent tests, and a case study on real data.

## 2. The case study

**Data.** Daily adjusted closing prices for SPY, the exchange-traded fund tracking the
S&P 500, from January 2008 to September 2026 — 4,411 trading days after warm-up,
17.5 years.

**The strategy family.** A moving-average crossover: go long when a short-window
average of price sits above a long-window one, short when below. It is a trend-following
rule, the most common idea in retail technical trading, and its two window lengths are
exactly the kind of parameters people tune. Fast windows run 2–60 days in steps of 2 and
slow windows 20–300 in steps of 10. Keeping only pairs where the fast window is shorter
gives **815 variants**.

Four design choices keep the backtest honest, and each is a way backtests commonly lie:

- **No look-ahead.** A signal computed from today's close can only be traded tomorrow,
  so every position is shifted one day. A test changes the final price in the series and
  asserts that no earlier position moves; if the future can reach back into the past, the
  test fails.
- **Transaction costs.** One basis point is charged per unit of position traded.
  Fast-window variants trade constantly and are the ones a parameter search is drawn to,
  so omitting costs would bias the search toward exactly the wrong answers.
- **Long/short rather than long/flat.** A long-or-flat rule spends most of its time long
  and quietly collects the market's own upward drift. Long/short holds little net market
  exposure over time, so what remains is the rule's own contribution.
- **A common date range.** A 300-day window cannot trade until day 301. Every variant is
  trimmed to the same dates so their Sharpe ratios are comparable.

## 3. The null world

Before touching real prices, the project builds a world where the answer is known:
families of strategies whose true expected return is exactly zero. Anything they appear
to earn is luck by construction, so this is the benchmark every real result is judged
against.

Take $N$ independent worthless strategies whose Sharpe estimates are spread with standard
deviation $\sigma_{SR}$. The expected best of them is, to a good approximation
(Bailey & López de Prado, 2014),

$$
E[\max SR] \approx \sigma_{SR}\left[(1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac1N\right) + \gamma\,\Phi^{-1}\!\left(1-\tfrac1{Ne}\right)\right]
$$

where $\gamma \approx 0.5772$ is the Euler–Mascheroni constant and $\Phi^{-1}$ is the
inverse of the standard normal distribution function. The formula comes from extreme
value theory, the branch of probability that describes the maxima of large samples. Its
key property is that it grows like $\sqrt{2\ln N}$: slowly, but without limit.

With five years of daily data and $N = 1{,}000$, the expected best annualised Sharpe is
**1.4557** by the formula. Simulating the same thing directly — generate a thousand
zero-edge strategies, take the best, repeat 200 times — gives **1.4600**. The two agree
to within the simulation's own sampling error. That check matters: it validates both the
formula and the code before either is trusted with real data.

## 4. The Sharpe ratio as an estimator

A measured Sharpe ratio is one draw from a distribution. Replay the same years with
different luck and you would measure a different number. The **standard error** is the
typical size of that wobble.

For returns with skewness $\gamma_3$ and kurtosis $\gamma_4$ (3 for a normal
distribution), the variance of the per-period Sharpe estimator over $T$ observations is
approximately $V/(T-1)$, where

$$
V = 1 - \gamma_3\,\widehat{SR} + \frac{\gamma_4 - 1}{4}\,\widehat{SR}^2 .
$$

This is Lo's (2002) result generalised by Mertens (2002). It comes from the delta
method: the 1 is the cost of estimating the mean, the skewness term is the covariance
between the estimates of mean and volatility, and the kurtosis term is the noisiness of
the volatility estimate itself.

Two consequences run through the rest of the project.

**The standard error of an annualised Sharpe is about $1/\sqrt{\text{years}}$,
regardless of how often you sample.** Daily, weekly or monthly data over five years all
give roughly ±0.45. Pinning a Sharpe down to ±0.1 takes about a century.

**Skewness matters much more than kurtosis for daily data.** Kurtosis enters multiplied
by $\widehat{SR}^2$, and per-period Sharpe ratios are small. The winning SPY variant has
kurtosis 14.2 — very fat-tailed — and skewness −0.97, yet together they widen the
standard error by only 1.3%, with the skewness term eleven times larger than the
kurtosis term.

From the standard error come two tools:

- The **Probabilistic Sharpe Ratio**, $\mathrm{PSR}(SR_0) = \Phi\big((\widehat{SR} - SR_0)/\mathrm{SE}\big)$:
  the probability that the true Sharpe exceeds a chosen benchmark $SR_0$. It is a
  one-sided $z$-test reported as a probability.
- The **Minimum Track Record Length**: the same expression solved for $T$, giving how
  many observations would be needed before a result of this size could be believed at a
  chosen confidence. It grows as $1/(\widehat{SR} - SR_0)^2$, so halving your margin over
  the benchmark quadruples the data required.

## 5. The Deflated Sharpe Ratio, and what calibration revealed

The **Deflated Sharpe Ratio** is the PSR with the benchmark set to the noise threshold
from section 3. Instead of asking whether the winner beats zero, it asks whether the
winner beats the best that $N$ worthless tries would produce. Only the spread of the trial
Sharpes enters the threshold, not their mean: the null hypothesis is that every trial has
zero true edge, so crediting the family with its observed average would assume the thing
under test.

Before using it, the project checks how it behaves when the answer is known. A
well-behaved probability computed under a true null should be spread evenly between 0
and 1. Running the DSR on 1,000 families of 100 zero-edge strategies each:

| | Undeflated PSR vs zero | Deflated Sharpe Ratio |
|---|---|---|
| Mean under the null | 0.9905 | 0.4950 |
| Fraction above 0.95 | **99.4%** | **0 of 1,000** |
| 5th / 95th percentile | — | 0.289 / 0.748 |

The first column is why this project exists. Measured the usual way, pure noise is
declared significant 994 times in 1,000. Deflation removes that entirely.

The second column contains a finding the literature's framing does not prepare you for:
**the DSR is not a p-value.** It is centred correctly but far too narrow — a
Kolmogorov–Smirnov test rejects uniformity at $p \approx 10^{-53}$. The cause is visible
in the diagnostics. The DSR divides by the standard error of a *single* strategy's
Sharpe (0.02822 per period here), but the *maximum* of many Sharpe ratios varies much
less from sample to sample (0.01168). The yardstick is 2.42 times too wide, so every
$z$-score is shrunk and the distribution collapses toward one half.

The implementation is correct — the tests pin it against Lo's closed form and against
brute-force simulation. The DSR simply answers a different question from a p-value: it
asks whether *this strategy's* true Sharpe clears the bar, allowing for uncertainty in
this strategy's estimate, not how surprising a maximum this large would be. Three
practical consequences follow:

- The DSR is **conservative**. It rarely blesses noise, which is the right direction for
  a tool whose job is scepticism.
- A DSR of 0.75 is **not** "75% likely real". Under pure noise it is roughly a
  one-in-twenty outcome.
- Thresholds should be **calibrated by simulation** at the $N$ and $T$ of the actual
  study, not borrowed from conventions built for uniform statistics.

## 6. Applying it to SPY

| | |
|---|---|
| Winner | fast = 28 days, slow = 200 days |
| Winner's annualised Sharpe | 0.4131 |
| PSR vs zero | 0.9559 |
| Noise threshold for 815 trials (formula) | 0.3143 |
| Deflated Sharpe Ratio | 0.6584 |
| Minimum track record vs that threshold | 284 years |
| Median variant | 0.2092 |
| Buy and hold SPY | 0.6432 |

Measured against zero, the winner clears the conventional 95% bar. Measured against what
815 tries hand out for free, it clears the bar by 0.10 annualised, against a standard
error of about 0.24 — and confirming that margin would take 284 years of data.

Following section 5, the DSR is read against a null distribution built at the same
$N$ and $T$: 300 families of 815 independent zero-edge strategies over 4,411 days. The
observed 0.6584 sits at the **85th percentile**; about **15%** of no-edge families do at
least as well. That verdict is revised in section 8.

The winner also loses to buy and hold. The comparison is not quite like for like — a
long/short strategy carries little market exposure, so an uncorrelated stream with a
Sharpe of 0.41 has some value inside a portfolio — but as a claim to have found
something, it does not stand.

## 7. Cross-validation: the Probability of Backtest Overfitting

The DSR is parametric: it relies on a model of how a Sharpe estimate is distributed,
and on a trial count that someone has to supply. The **Probability of Backtest
Overfitting** (Bailey, Borwein, López de Prado & Zhu) uses neither.

**Combinatorially symmetric cross-validation** splits the history into 16 blocks of 275
days and takes every way of choosing 8 of them as in-sample — $\binom{16}{8} = 12{,}870$
splits. In each split it picks the variant that did best in sample, then looks up where
that same variant ranks among all 815 on the other 8 blocks. The method is symmetric
because every block serves as training and test data equally often, so no single
train/test cut can be chosen to flatter the answer. PBO is the fraction of splits in
which the in-sample winner finishes in the bottom half out of sample.

| | |
|---|---|
| Probability of Backtest Overfitting | **0.898** |
| Median winner's Sharpe, in sample | +0.60 |
| Median winner's Sharpe, out of sample | −0.02 |
| Probability of an out-of-sample loss | 0.545 |
| Degradation slope, out-of-sample on in-sample | −0.56 |

Pure noise would give a PBO near 0.5. At 0.9, selection is not merely uninformative but
actively misleading: picking the best-looking variant is worse than picking one at
random. This is the **winner's curse** — the maximum of many noisy estimates
preferentially selects the most favourable error, and errors do not persist. The
negative slope says the same thing another way: the better a winner looked in sample, the
worse it did afterwards.

Done naively, the computation is about 46 billion operations. It runs in 0.4 seconds
because a Sharpe ratio needs only a count, a sum and a sum of squares, and all three add
up across blocks. Computing them once per block turns all 12,870 splits into a single
matrix multiplication.

## 8. How many trials were there?

The DSR's threshold needs $N$ independent trials. A parameter sweep does not supply
them: a 28/200 crossover and a 30/200 crossover hold nearly the same position almost
every day.

**Counting by structure gives the wrong answer.** By the participation ratio of the
correlation matrix's eigenvalues, the 815 variants span about **2.7** independent
directions; hierarchical clustering at a 0.5 correlation threshold finds **3** groups.
Plugging $N \approx 3$ into the deflation lifts the DSR from 0.66 to about **0.91** and
makes the strategy look respectable. That reasoning is wrong. These measures count
independent directions in *return space*, not independent *chances for luck*: the
maximum was still taken over 815 noisy realisations.

**Measuring the null directly.** The operational question is what the best of *these*
815 variants looks like on data with nothing to exploit. The project resamples SPY's
daily returns in blocks of 21 days — which destroys trends and reversals while keeping
the way calm and volatile periods cluster — rebuilds a price path, and reruns the whole
sweep. Repeated 300 times, this measures luck for the actual family, with its variants
exactly as correlated as they really are and no independence assumption anywhere.

| | Formula (N = 815 independent) | Measured (block bootstrap) |
|---|---|---|
| Expected best Sharpe | 0.31 | **0.59** |
| Chance a no-edge history beats the real winner | 0.15 | **0.82** |

The measured bar is almost twice the formula's. Two effects compound:

- The spread of trial Sharpe ratios is about **46% wider** on structureless data
  (0.144 annualised) than on real SPY (0.098). SPY's long bull market pushes most
  variants the same way and compresses their spread; remove the trend and each variant's
  own trading pattern dominates.
- Even allowing for that wider spread, the best variant exceeds the Gaussian
  extreme-value prediction by **17–28%**, depending on block length. The spread of these
  Sharpe ratios is not Gaussian: fast variants bleed costs, slow ones barely trade.

The probability is stable at **0.70–0.82** across block lengths from 1 to 63 days.

One number is deliberately not reported. Inverting the measured maximum into an
"implied number of independent trials" gives about 22,000 with 21-day blocks and about
7,000 with a plain shuffle. Because the expected maximum grows like $\sqrt{2\ln N}$, a
7% change in the measured maximum moves the implied $N$ by a factor of three. The
bootstrap probability needs no such inversion.

Enabling all of this was an optimisation: the original sweep recomputed each moving
average dozens of times. Computing each window length once gives identical output —
a test asserts it — about 17 times faster, which is what made 300 complete reruns
affordable.

## 9. What changed my mind

**The direction of the correlation correction.** After seeing how smooth the Sharpe
heatmap was, I expected correlated variants to mean fewer effective trials and so an
easier bar. Measurement showed the formula's bar was too *low*. Had I trusted either the
intuition or the structural counts, the verdict would have moved the wrong way.

**Reading DSR as a probability.** The name invites reading 0.66 as "66% likely real".
The calibration study showed it is not uniform under the null, which is what made the
matched-null and bootstrap comparisons necessary rather than optional.

**A test caught me in the project's own trap.** To check that a genuine edge survives
deflation, I planted one strategy with a per-period edge of 0.05 among 30 pure-noise
strategies. It lost. The best of 30 noise strategies over 800 days reaches about 0.069,
so a real but modest edge was beaten by luck — exactly the effect the tool exists to
measure, inside the code written to measure it.

## 10. Limitations

- **One asset, one family.** Everything rests on SPY and moving-average crossovers.
  Trend rules on a single large equity index are close to the most-studied strategies
  there are.
- **The true trial count is unknowable.** Every choice in this study — SPY, 2008, this
  rule, this grid — was itself selected, and the unrecorded alternatives count as trials
  too. Many other people have swept this same grid, which raises the effective count
  further in a way nobody records. The bootstrap accounts for the 815 variants swept
  here, not for the search that led to them.
- **Costs are simplified.** A flat one basis point per trade, with no market impact,
  slippage or cost of borrowing to short. Realistic costs would lower every variant's
  Sharpe and strengthen the conclusion.
- **The bootstrap's null is a model.** Resampling in fixed 21-day blocks assumes the
  return process is stable over time and that block length captures its dependence. The
  answer is stable across block lengths, but a regime-switching market is not fully
  represented.
- **CSCV's splits are not independent.** The 12,870 splits overlap heavily, so PBO is a
  well-defined proportion, not a sample of 12,870 independent trials, and carries no
  simple standard error.
- **The data are from Yahoo Finance**, via the `yfinance` library. Prices are cached and
  bundled for reproducibility, not audited.

## 11. What I would do next

- **More assets and families.** Run the same pipeline across equity indices, bonds,
  commodities and currencies, and across rule families, to see whether the conclusion is
  specific to SPY or general.
- **A positive control on real data.** Every result here is negative. Planting a known
  edge into real returns and checking that all three tests detect it would show the
  tools can say yes as well as no.
- **Multiple-testing corrections.** Compare against Harvey and Liu's "haircut" Sharpe
  ratio, which approaches the same problem through adjustments for testing many
  hypotheses at once.
- **A better bootstrap.** The stationary bootstrap of Politis and Romano draws random
  block lengths, removing the one tuning choice left in section 8.
- **Walk-forward testing.** Choose parameters on a rolling past window and trade them on
  the next period, repeatedly. This mimics how a strategy would actually be run.

## 12. Reproducing everything

Every simulation uses a fixed random seed, so the figures above reproduce exactly.

```bash
uv sync
uv run pytest -q                                # 44 tests
uv run python -m scripts.fig_null_world         # section 3
uv run python -m scripts.demo_sharpe            # section 4
uv run python -m scripts.fig_dsr_calibration    # section 5
uv run python -m scripts.run_sweep              # section 6
uv run python -m scripts.null_reference         # section 6, matched null
uv run python -m scripts.run_pbo                # section 7
uv run python -m scripts.fig_effective_trials   # section 8
uv run streamlit run app.py                     # the dashboard
```

## References

- Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio
  Management*. [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. The Probability of
  Backtest Overfitting. *Journal of Computational Finance*.
  [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*.
- Mertens, E. (2002). Comments on the variance of the IID estimator in Lo (2002).
  Working paper.
- Politis, D. N. & Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the
  American Statistical Association*.
