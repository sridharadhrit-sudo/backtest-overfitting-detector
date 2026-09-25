"""Checks on combinatorially symmetric cross-validation."""

import numpy as np
import pytest
from math import comb

from src.nulls import simulate_null_returns
from src.pbo import block_moments, cscv


def test_split_count_is_the_binomial_coefficient():
    rng = np.random.default_rng(1)
    result = cscv(simulate_null_returns(20, 640, rng), n_blocks=10)
    assert result["n_splits"] == comb(10, 5)


def test_block_moments_reconstruct_the_sharpe():
    """Sums and sums of squares must rebuild exactly what a direct calculation gives."""
    rng = np.random.default_rng(2)
    matrix = simulate_null_returns(5, 800, rng)
    sums, sqs, block_len = block_moments(matrix, n_blocks=8)
    n = block_len * 8
    mean = sums.sum(axis=1) / n
    var = (sqs.sum(axis=1) - sums.sum(axis=1) ** 2 / n) / (n - 1)
    direct = matrix[:, : n].mean(axis=1) / matrix[:, : n].std(axis=1, ddof=1)
    assert np.allclose(mean / np.sqrt(var), direct)


def test_pure_noise_gives_pbo_near_a_half():
    """With no edge anywhere, the in-sample winner is a coin flip out of sample."""
    rng = np.random.default_rng(3)
    result = cscv(simulate_null_returns(60, 1600, rng), n_blocks=10)
    assert result["pbo"] == pytest.approx(0.5, abs=0.2)


def test_a_genuine_edge_drives_pbo_to_zero():
    """A strategy that really is best should keep winning out of sample."""
    rng = np.random.default_rng(4)
    matrix = simulate_null_returns(40, 1600, rng)
    matrix[11] += 0.35
    result = cscv(matrix, n_blocks=10)
    assert result["pbo"] < 0.02


def test_logit_sign_matches_the_upper_half():
    """A positive logit must mean the winner landed in the upper half out of sample."""
    rng = np.random.default_rng(5)
    result = cscv(simulate_null_returns(30, 1200, rng), n_blocks=10)
    assert np.isclose(result["pbo"], (result["logits"] <= 0).mean())


def test_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(6)
    matrix = simulate_null_returns(25, 1200, rng)
    small = cscv(matrix, n_blocks=10, chunk=7)
    large = cscv(matrix, n_blocks=10, chunk=10_000)
    assert small["pbo"] == large["pbo"]
    assert np.allclose(small["logits"], large["logits"])


def test_too_many_blocks_is_rejected():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        cscv(simulate_null_returns(5, 10, rng), n_blocks=16)
