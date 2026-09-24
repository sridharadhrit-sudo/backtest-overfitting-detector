"""The Deflated Sharpe Ratio.

DSR is the Probabilistic Sharpe Ratio with the benchmark set to the Sharpe that the
best of N worthless strategies would reach by luck alone. It asks not "is this better
than nothing?" but "is this better than what searching N times hands out for free?"

Bailey & Lopez de Prado (2014).
"""

import numpy as np

from src.nulls import expected_max_sharpe, sharpe_ratios
from src.sharpe import moments, probabilistic_sharpe_ratio, sharpe_ratio


def deflation_threshold(trial_sharpes, n_trials=None):
    """SR_0: the Sharpe luck alone produces as the best of `n_trials`.

    The null hypothesis is that every trial has a TRUE Sharpe of zero, so only the
    cross-sectional DISPERSION of the observed trial Sharpes is used, not their mean.
    Adding the mean back in would assume the strategies have real edge, which is
    exactly what we are trying to test.
    """
    s = np.asarray(trial_sharpes, dtype=float)
    n = s.size if n_trials is None else n_trials
    if n < 2:
        return 0.0
    return expected_max_sharpe(n, s.std(ddof=1))


def deflated_sharpe_ratio(winner_returns, sigma_sr, n_trials):
    """P(the winner's true Sharpe beats the N-trial noise threshold)."""
    r = np.asarray(winner_returns, dtype=float)
    sr = sharpe_ratio(r)
    g3, g4 = moments(r)
    sr0 = expected_max_sharpe(n_trials, sigma_sr)
    return probabilistic_sharpe_ratio(sr, r.size, sr0, g3, g4)


def deflate_family(returns_matrix, n_trials=None):
    """Select the in-sample winner from a family of strategies and deflate it.

    `returns_matrix` is (n_strategies, n_periods): one row per strategy variant.
    `n_trials` defaults to the number of rows, but should be the EFFECTIVE number of
    independent trials once Stage 6 exists -- a grid sweep is not N independent tries.
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    trial_srs = sharpe_ratios(matrix)
    winner = int(np.argmax(trial_srs))
    n = matrix.shape[0] if n_trials is None else n_trials

    sigma_sr = trial_srs.std(ddof=1)
    winner_returns = matrix[winner]
    g3, g4 = moments(winner_returns)
    sr = trial_srs[winner]
    sr0 = expected_max_sharpe(n, sigma_sr)

    return {
        "winner_index": winner,
        "sharpe": sr,
        "threshold": sr0,
        "sigma_sr": sigma_sr,
        "n_trials": n,
        "skewness": g3,
        "kurtosis": g4,
        "dsr": probabilistic_sharpe_ratio(sr, winner_returns.size, sr0, g3, g4),
        "psr_vs_zero": probabilistic_sharpe_ratio(sr, winner_returns.size, 0.0, g3, g4),
    }
