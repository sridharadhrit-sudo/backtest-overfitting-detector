"""Probability of Backtest Overfitting, via combinatorially symmetric cross-validation.

A second, non-parametric attack on the same question the DSR asks. Instead of modelling
the uncertainty in one Sharpe ratio, it tests the SELECTION PROCEDURE itself: if I pick
the best variant on half the history, does it still work on the other half?

Symmetric because every block serves as both training and test data across the full set
of splits, so no arbitrary train/test cut can be chosen favourably.

Bailey, Borwein, Lopez de Prado & Zhu (2015).
"""

from itertools import combinations

import numpy as np

DEFAULT_BLOCKS = 16


def block_moments(returns_matrix, n_blocks=DEFAULT_BLOCKS):
    """Per-block sums and sums of squares for every strategy.

    The whole method rests on this: a Sharpe ratio needs only a count, a sum and a sum
    of squares, and all three are additive across blocks. So the 12,870 splits can be
    assembled from 16 precomputed pieces instead of re-scanning the returns each time.
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    n_strategies, n_periods = matrix.shape
    block_len = n_periods // n_blocks
    if block_len < 2:
        raise ValueError("not enough periods for that many blocks")
    usable = matrix[:, : block_len * n_blocks]
    reshaped = usable.reshape(n_strategies, n_blocks, block_len)
    return reshaped.sum(axis=2), (reshaped ** 2).sum(axis=2), block_len


def _sharpes_from_moments(total, total_sq, n_obs):
    """Sharpe ratios assembled from sums, for a whole chunk of splits at once."""
    mean = total / n_obs
    variance = (total_sq - total ** 2 / n_obs) / (n_obs - 1)
    return mean / np.sqrt(np.maximum(variance, 1e-300))


def cscv(returns_matrix, n_blocks=DEFAULT_BLOCKS, chunk=512):
    """Run combinatorially symmetric cross-validation.

    Returns a dict with the logit of each split's out-of-sample rank, the PBO, the
    in-sample and out-of-sample Sharpe of every split's winner, and summary statistics.
    """
    sums, sqs, block_len = block_moments(returns_matrix, n_blocks)
    n_strategies = sums.shape[0]
    half = n_blocks // 2
    n_obs = block_len * half

    splits = list(combinations(range(n_blocks), half))
    logits = np.empty(len(splits))
    is_sr = np.empty(len(splits))
    oos_sr = np.empty(len(splits))

    for start in range(0, len(splits), chunk):
        block = splits[start : start + chunk]
        indicator = np.zeros((len(block), n_blocks))
        for row, combo in enumerate(block):
            indicator[row, list(combo)] = 1.0
        complement = 1.0 - indicator

        sr_in = _sharpes_from_moments(indicator @ sums.T, indicator @ sqs.T, n_obs)
        sr_out = _sharpes_from_moments(complement @ sums.T, complement @ sqs.T, n_obs)

        winner = np.argmax(sr_in, axis=1)
        rows = np.arange(len(block))
        winner_oos = sr_out[rows, winner]

        # Rank of the winner among all strategies out of sample: 1 = worst, N = best.
        rank = (sr_out < winner_oos[:, None]).sum(axis=1) + 1
        omega = rank / (n_strategies + 1.0)
        logits[start : start + len(block)] = np.log(omega / (1.0 - omega))
        is_sr[start : start + len(block)] = sr_in[rows, winner]
        oos_sr[start : start + len(block)] = winner_oos

    slope, intercept = np.polyfit(is_sr, oos_sr, 1)

    return {
        "n_splits": len(splits),
        "n_blocks": n_blocks,
        "n_strategies": n_strategies,
        "block_length": block_len,
        "logits": logits,
        "is_sharpe": is_sr,
        "oos_sharpe": oos_sr,
        "pbo": float((logits <= 0).mean()),
        "prob_oos_loss": float((oos_sr < 0).mean()),
        "median_oos_sharpe": float(np.median(oos_sr)),
        "degradation_slope": float(slope),
        "degradation_intercept": float(intercept),
    }
