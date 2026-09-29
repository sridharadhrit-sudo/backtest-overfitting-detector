"""How many of the trials were actually independent?

The E[max SR] formula wants N INDEPENDENT trials. A parameter sweep supplies nothing of
the sort: a 28/200 crossover and a 30/200 crossover hold nearly the same position nearly
every day. Counting them as two separate attempts sets the bar far too high.

Two ways to count honestly, with different weaknesses:

  clusters  -- group variants whose returns correlate above a threshold, count groups.
               Interpretable, but the threshold is a judgement call.
  spectral  -- the participation ratio of the correlation matrix's eigenvalues,
               (sum of eigenvalues)^2 / (sum of squares). No threshold to choose, but
               harder to explain and it counts directions rather than strategies.

Neither is "the" answer, which is why the deliverable is a sensitivity curve rather
than a single number.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def correlation_matrix(returns_matrix):
    """Correlation between every pair of strategies' return series."""
    return np.corrcoef(np.asarray(returns_matrix, dtype=float))


def correlation_distance(corr):
    """Turn correlations into a proper metric: d = sqrt((1 - rho) / 2).

    Identical series are 0 apart, uncorrelated ones sqrt(1/2), perfectly opposed ones 1.
    This is the standard choice because it satisfies the triangle inequality, which
    hierarchical clustering assumes.
    """
    return np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))


def cluster_count(returns_matrix, min_correlation=0.5, method="average"):
    """Number of groups once variants correlating above `min_correlation` are merged."""
    corr = correlation_matrix(returns_matrix)
    distance = correlation_distance(corr)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method=method)
    cutoff = np.sqrt((1.0 - min_correlation) / 2.0)
    return int(fcluster(tree, t=cutoff, criterion="distance").max())


def spectral_effective_trials(returns_matrix):
    """Participation ratio of the correlation matrix's eigenvalues.

    Equals n when the strategies are mutually uncorrelated and 1 when they are all the
    same strategy, so it reads as an effective count of independent directions.
    """
    eigenvalues = np.linalg.eigvalsh(correlation_matrix(returns_matrix))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


def cluster_curve(returns_matrix, thresholds=None, method="average"):
    """Cluster count across a range of correlation thresholds.

    Showing the whole curve is more honest than picking one cutoff and reporting it as
    though it were measured rather than chosen.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 1.0, 0.05), 2)
    corr = correlation_matrix(returns_matrix)
    distance = correlation_distance(corr)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method=method)
    counts = [
        int(fcluster(tree, t=np.sqrt((1.0 - rho) / 2.0), criterion="distance").max())
        for rho in thresholds
    ]
    return np.asarray(thresholds), np.asarray(counts)


def bootstrap_family_null(prices, fast_windows, slow_windows, n_bootstrap=300,
                          block_length=21, cost=None, seed=0, on_step=None):
    """The null distribution of the winner's Sharpe, for THIS family of strategies.

    Resampling the daily returns in blocks destroys trends and reversals -- anything a
    moving-average rule could exploit -- while a block length of about a month keeps
    volatility clustering intact, so the synthetic path still behaves like a market.
    Rebuild a price path, rerun the whole sweep, record the best Sharpe, repeat.

    The variants stay exactly as correlated with one another as they really are, because
    they are the same strategies applied to the same kind of path. So this measures what
    "the best of my 815 variants, by luck alone" actually means, with no assumption that
    the trials were independent. Results are stable for block lengths from 1 to 63.

    Set block_length=1 for a plain i.i.d. shuffle. `on_step(done, total)`, if given,
    is called after each resample so a caller can show progress.
    """
    from src.backtest import DEFAULT_COST, fast_sweep

    cost = DEFAULT_COST if cost is None else cost
    values = np.asarray(prices, dtype=float)
    returns = values[1:] / values[:-1] - 1.0
    start = values[0]

    rng = np.random.default_rng(seed)
    best = np.empty(n_bootstrap)
    dispersion = np.empty(n_bootstrap)
    n_blocks = int(np.ceil(returns.size / block_length))

    for i in range(n_bootstrap):
        if block_length <= 1:
            resampled = rng.permutation(returns)
        else:
            starts = rng.integers(0, returns.size - block_length, size=n_blocks)
            resampled = np.concatenate(
                [returns[s : s + block_length] for s in starts]
            )[: returns.size]
        path = np.concatenate(([start], start * np.cumprod(1.0 + resampled)))
        matrix, _ = fast_sweep(path, fast_windows, slow_windows, cost)
        sharpes = matrix.mean(axis=1) / matrix.std(axis=1, ddof=1)
        best[i] = sharpes.max()
        dispersion[i] = sharpes.std(ddof=1)
        if on_step is not None:
            on_step(i + 1, n_bootstrap)

    return best, dispersion


def implied_independent_trials(expected_max, sigma_sr, max_trials=100_000):
    """How many INDEPENDENT trials would give this expected maximum?

    Inverts the Bailey-Lopez de Prado expression numerically. This is the operational
    definition of an effective trial count: not a clustering heuristic, but the number
    of independent attempts that reproduces the family's own measured behaviour.
    """
    from src.nulls import expected_max_sharpe

    if sigma_sr <= 0:
        return 1.0
    target = expected_max / sigma_sr
    grid = np.unique(np.round(np.geomspace(2, max_trials, 4000)).astype(int))
    curve = np.array([expected_max_sharpe(int(n), 1.0) for n in grid])
    if target <= curve[0]:
        return float(grid[0])
    if target >= curve[-1]:
        return float(grid[-1])
    return float(np.interp(target, curve, grid))
