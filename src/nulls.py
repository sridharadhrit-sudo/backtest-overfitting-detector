"""The null world: strategies with no edge whatsoever."""

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329
TRADING_DAYS = 252


def simulate_null_returns(n_strategies, n_periods, rng):
    """Returns for `n_strategies` strategies whose true expected return is zero."""
    return rng.standard_normal((n_strategies, n_periods))


def sharpe_ratios(returns):
    """Per-period Sharpe ratio of each strategy (each row of `returns`)."""
    return returns.mean(axis=1) / returns.std(axis=1, ddof=1)


def expected_max_sharpe(n_trials, sigma_sr=1.0):
    """Expected best Sharpe from `n_trials` independent worthless strategies.

    Bailey & Lopez de Prado (2014), equation for E[max SR].
    """
    g = EULER_MASCHERONI
    return sigma_sr * (
        (1 - g) * norm.ppf(1 - 1 / n_trials)
        + g * norm.ppf(1 - 1 / (n_trials * np.e))
    )


def annualise(sharpe_per_period, periods_per_year=TRADING_DAYS):
    """Convert a per-period Sharpe to an annual one."""
    return sharpe_per_period * np.sqrt(periods_per_year)


def max_sharpe_distribution(n_strategies, n_periods, n_experiments, seed=0):
    """Repeat the whole experiment and collect the winner's Sharpe each time."""
    rng = np.random.default_rng(seed)
    winners = np.empty(n_experiments)
    for i in range(n_experiments):
        returns = simulate_null_returns(n_strategies, n_periods, rng)
        winners[i] = sharpe_ratios(returns).max()
    return winners
