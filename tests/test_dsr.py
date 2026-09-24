"""Checks on the Deflated Sharpe Ratio."""

import numpy as np
import pytest

from src.dsr import deflate_family, deflated_sharpe_ratio, deflation_threshold
from src.nulls import simulate_null_returns


def test_threshold_rises_with_trial_count():
    """Searching harder must raise the bar."""
    sharpes = np.array([0.01, -0.02, 0.03, 0.00, -0.01])
    assert deflation_threshold(sharpes, 10) < deflation_threshold(sharpes, 1000)


def test_threshold_scales_with_dispersion():
    """Twice the spread of trial Sharpes, twice the threshold."""
    tight = np.array([0.00, 0.01, -0.01, 0.02, -0.02])
    wide = tight * 2
    assert deflation_threshold(wide, 100) == pytest.approx(
        2 * deflation_threshold(tight, 100)
    )


def test_single_trial_needs_no_deflation():
    assert deflation_threshold(np.array([0.05]), 1) == 0.0


def test_deflation_only_ever_lowers_confidence():
    """DSR must never exceed the undeflated PSR."""
    rng = np.random.default_rng(2)
    family = simulate_null_returns(50, 1000, rng)
    report = deflate_family(family)
    assert report["dsr"] <= report["psr_vs_zero"]


def test_winner_is_the_argmax():
    rng = np.random.default_rng(5)
    family = simulate_null_returns(30, 800, rng)
    family[7] += 0.20  # plant an edge big enough to beat 30 tries of luck
    assert deflate_family(family)["winner_index"] == 7


def test_real_edge_survives_deflation():
    """A strategy with a genuinely large edge should clear the bar comfortably."""
    rng = np.random.default_rng(9)
    family = simulate_null_returns(100, 5 * 252, rng)
    family[0] += 0.20  # per-period Sharpe far above anything luck produces
    assert deflate_family(family)["dsr"] > 0.99


def test_null_families_average_a_coin_flip():
    """Across many no-edge families the mean DSR must sit near 0.5."""
    rng = np.random.default_rng(21)
    values = [deflate_family(simulate_null_returns(60, 1260, rng))["dsr"]
              for _ in range(120)]
    assert np.mean(values) == pytest.approx(0.5, abs=0.08)


def test_more_assumed_trials_lowers_dsr():
    """Honestly declaring more trials must make you less confident."""
    rng = np.random.default_rng(4)
    family = simulate_null_returns(80, 1260, rng)
    assert deflate_family(family, n_trials=10)["dsr"] > \
           deflate_family(family, n_trials=5000)["dsr"]


def test_deflated_sharpe_ratio_matches_family_helper():
    rng = np.random.default_rng(6)
    family = simulate_null_returns(40, 900, rng)
    report = deflate_family(family)
    direct = deflated_sharpe_ratio(
        family[report["winner_index"]], report["sigma_sr"], report["n_trials"]
    )
    assert direct == pytest.approx(report["dsr"])
