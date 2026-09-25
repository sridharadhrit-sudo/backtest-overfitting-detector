"""Checks on the strategy family. No network: everything runs on synthetic prices."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import crossover_positions, parameter_grid, strategy_returns, sweep
from src.data import synthetic_prices, to_returns


def test_position_uses_only_past_information():
    """The defining test: today's position must not depend on today's price.

    Change the final price arbitrarily. Every position up to and including the last day
    must be untouched -- if any of them moves, the backtest is peeking at the future.
    """
    prices = synthetic_prices(400, seed=1)
    tampered = prices.copy()
    tampered.iloc[-1] *= 1.5

    base = crossover_positions(prices, 5, 20)
    peeked = crossover_positions(tampered, 5, 20)
    pd.testing.assert_series_equal(base, peeked)


def test_costs_only_ever_reduce_returns():
    prices = synthetic_prices(600, seed=2)
    free = strategy_returns(prices, 10, 50, cost=0.0)
    costly = strategy_returns(prices, 10, 50, cost=0.001)
    assert costly.sum() < free.sum()


def test_costs_are_charged_only_when_trading():
    """A day with no change of position must cost nothing."""
    prices = synthetic_prices(600, seed=3)
    free = strategy_returns(prices, 10, 50, cost=0.0)
    costly = strategy_returns(prices, 10, 50, cost=0.01)
    position = crossover_positions(prices, 10, 50)
    unchanged = position.diff().fillna(0.0) == 0
    assert np.allclose(free[unchanged].dropna(), costly[unchanged].dropna())


def test_positions_are_long_or_short_only():
    prices = synthetic_prices(500, seed=4)
    held = crossover_positions(prices, 8, 40).dropna()
    assert set(np.unique(held)).issubset({-1.0, 0.0, 1.0})


def test_fast_must_be_shorter_than_slow():
    prices = synthetic_prices(200, seed=5)
    with pytest.raises(ValueError):
        crossover_positions(prices, 50, 20)


def test_grid_excludes_invalid_pairs():
    pairs = parameter_grid([5, 10, 20], [10, 20, 50])
    assert all(f < s for f, s in pairs)
    assert (10, 10) not in pairs


def test_sweep_shapes_line_up():
    prices = synthetic_prices(1500, seed=6)
    matrix, params = sweep(prices, [5, 10], [50, 100])
    assert matrix.shape[0] == len(params)
    assert not np.isnan(matrix).any()


def test_sweep_variants_share_a_date_range():
    """Every row must cover the same days, or their Sharpes are not comparable."""
    prices = synthetic_prices(1500, seed=7)
    matrix, _ = sweep(prices, [5, 20], [60, 200])
    assert len({row.size for row in matrix}) == 1


def test_synthetic_prices_have_no_exploitable_edge():
    """Sanity check on the test fixture itself: it is a random walk, nothing more."""
    returns = to_returns(synthetic_prices(4000, seed=8))
    autocorr = returns.autocorr(lag=1)
    assert abs(autocorr) < 0.06
