"""Checks on the effective-trial-count machinery."""

import numpy as np
import pytest

from src.effective_trials import (
    cluster_count,
    cluster_curve,
    correlation_distance,
    implied_independent_trials,
    spectral_effective_trials,
)
from src.nulls import expected_max_sharpe, simulate_null_returns


def test_correlation_distance_endpoints():
    corr = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    distance = correlation_distance(corr)
    assert distance[0, 0] == pytest.approx(0.0)
    assert distance[0, 1] == pytest.approx(np.sqrt(0.5))
    assert distance[0, 2] == pytest.approx(1.0)


def test_identical_strategies_collapse_to_one():
    """Twenty copies of the same thing are one trial, not twenty."""
    rng = np.random.default_rng(1)
    single = rng.standard_normal(600)
    family = np.tile(single, (20, 1))
    assert cluster_count(family, min_correlation=0.9) == 1
    assert spectral_effective_trials(family) == pytest.approx(1.0, abs=1e-6)


def test_independent_strategies_count_themselves():
    rng = np.random.default_rng(2)
    family = simulate_null_returns(40, 4000, rng)
    assert spectral_effective_trials(family) > 30


def test_spectral_measure_sits_between_one_and_n():
    rng = np.random.default_rng(3)
    family = simulate_null_returns(25, 800, rng)
    family[10:] = family[0] + 0.05 * family[10:]   # make most of them near-copies
    value = spectral_effective_trials(family)
    assert 1.0 <= value <= 25.0


def test_cluster_curve_is_monotone():
    """Demanding tighter correlation to merge can only leave more groups."""
    rng = np.random.default_rng(4)
    family = simulate_null_returns(30, 900, rng)
    _, counts = cluster_curve(family, thresholds=[0.2, 0.4, 0.6, 0.8])
    assert list(counts) == sorted(counts)


def test_implied_trials_inverts_the_formula():
    """Feed in the expected maximum for a known N and get that N back."""
    for n in [10, 100, 1000]:
        sigma = 0.03
        implied = implied_independent_trials(expected_max_sharpe(n, sigma), sigma)
        assert implied == pytest.approx(n, rel=0.05)


def test_implied_trials_handles_degenerate_dispersion():
    assert implied_independent_trials(0.1, 0.0) == 1.0
