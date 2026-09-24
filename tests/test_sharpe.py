"""Checks on the Sharpe estimator machinery."""

import numpy as np
import pytest
from scipy.stats import norm

from src.sharpe import (
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_std_error,
    sharpe_variance_factor,
)
from src.units import annualise, deannualise


def test_variance_factor_collapses_to_lo():
    """With gamma_3 = 0 and gamma_4 = 3 the bracket must equal 1 + SR^2/2."""
    for sr in [0.0, 0.05, 0.2, 1.0]:
        assert sharpe_variance_factor(sr) == pytest.approx(1 + sr ** 2 / 2)


def test_std_error_matches_simulation():
    """The analytic standard error must match the spread of simulated estimates."""
    rng = np.random.default_rng(7)
    n_periods, n_trials = 1000, 20_000
    draws = rng.standard_normal((n_trials, n_periods))
    estimates = draws.mean(axis=1) / draws.std(axis=1, ddof=1)
    assert estimates.std(ddof=1) == pytest.approx(sharpe_std_error(0.0, n_periods), rel=0.03)


def test_psr_is_half_at_the_benchmark():
    """A Sharpe exactly equal to the benchmark is a coin flip, whatever the sample."""
    assert probabilistic_sharpe_ratio(0.1, 500, benchmark=0.1) == pytest.approx(0.5)


def test_psr_rises_with_sample_length():
    """The same Sharpe is more believable from more data."""
    short = probabilistic_sharpe_ratio(0.06, 250)
    long = probabilistic_sharpe_ratio(0.06, 2500)
    assert short < long


def test_negative_skew_and_fat_tails_widen_the_error():
    """Both make the same Sharpe less trustworthy, so the error must grow."""
    base = sharpe_std_error(0.1, 1000)
    assert sharpe_std_error(0.1, 1000, gamma_3=-1.5) > base
    assert sharpe_std_error(0.1, 1000, gamma_4=9.0) > base


def test_min_track_record_round_trips():
    """Feeding MinTRL back into PSR must return the confidence level asked for."""
    sr, benchmark, confidence = 0.08, 0.02, 0.95
    n = min_track_record_length(sr, benchmark, confidence=confidence)
    assert probabilistic_sharpe_ratio(sr, n, benchmark) == pytest.approx(confidence)


def test_min_track_record_infinite_below_benchmark():
    assert min_track_record_length(0.01, benchmark=0.05) == np.inf


def test_annualised_std_error_is_one_over_root_years():
    """The headline result: SE of an annualised Sharpe depends only on calendar span."""
    for years in [1, 3, 5, 20]:
        n_periods = years * 252
        se = annualise(sharpe_std_error(0.0, n_periods))
        assert se == pytest.approx(1 / np.sqrt(years), rel=0.01)


def test_units_round_trip():
    assert deannualise(annualise(0.0731)) == pytest.approx(0.0731)


def test_sharpe_ratio_recovers_a_known_value():
    """A series with known mean and sd must give the ratio of the two."""
    rng = np.random.default_rng(3)
    r = rng.normal(loc=0.001, scale=0.01, size=200_000)
    assert sharpe_ratio(r) == pytest.approx(0.1, rel=0.05)
