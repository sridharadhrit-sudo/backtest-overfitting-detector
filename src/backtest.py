"""A parameterised strategy family, swept over a grid, with costs and no look-ahead.

The rule: go long when a fast moving average is above a slow one, short when below.
Long/short rather than long/flat, so the strategy cannot inherit the market's own
upward drift and pass it off as skill.
"""

import numpy as np
import pandas as pd

DEFAULT_COST = 0.0001  # 1 basis point of notional per unit of position traded


def crossover_positions(prices, fast, slow):
    """Position held on each day, using only information available the day before.

    The signal computed from closes up to and including day t can only be traded from
    day t+1. The .shift(1) IS that rule. Removing it produces beautiful, fictional
    equity curves -- this is look-ahead bias, and it is the most common backtest bug.
    """
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal = np.sign(fast_ma - slow_ma)
    return signal.shift(1)


def strategy_returns(prices, fast, slow, cost=DEFAULT_COST):
    """Net daily returns of one variant, after transaction costs."""
    asset_returns = prices.pct_change()
    position = crossover_positions(prices, fast, slow)
    turnover = position.diff().abs().fillna(0.0)
    return (position * asset_returns - cost * turnover).rename(f"ma_{fast}_{slow}")


def parameter_grid(fast_windows, slow_windows):
    """All (fast, slow) pairs with fast strictly shorter than slow."""
    return [(f, s) for f in fast_windows for s in slow_windows if f < s]


def sweep(prices, fast_windows, slow_windows, cost=DEFAULT_COST):
    """Backtest every variant on a common date range.

    Returns (matrix, params): matrix is (n_variants, n_days), params the (fast, slow)
    pairs in matching order. All variants share the same dates, so their Sharpe ratios
    are comparable and the cross-sectional dispersion means something.
    """
    params = parameter_grid(fast_windows, slow_windows)
    if not params:
        raise ValueError("no valid (fast, slow) pairs in the supplied windows")

    warmup = max(s for _, s in params) + 1
    columns = {f"{f}_{s}": strategy_returns(prices, f, s, cost) for f, s in params}
    frame = pd.DataFrame(columns).iloc[warmup:]
    frame = frame.dropna(how="any")
    return frame.to_numpy().T, params


def fast_sweep(prices, fast_windows, slow_windows, cost=DEFAULT_COST):
    """Same answer as `sweep`, but built from precomputed rolling means.

    `sweep` recomputes both moving averages for every one of the 815 pairs, so each
    window length is recalculated dozens of times. Here every distinct window is
    averaged once and the pairs are assembled from the results. Same output, roughly
    two orders of magnitude faster -- which is what makes the bootstrap in Stage 6
    affordable.
    """
    params = parameter_grid(fast_windows, slow_windows)
    if not params:
        raise ValueError("no valid (fast, slow) pairs in the supplied windows")

    values = np.asarray(prices, dtype=float)
    asset_returns = np.empty_like(values)
    asset_returns[0] = np.nan
    asset_returns[1:] = values[1:] / values[:-1] - 1.0

    windows = sorted({w for pair in params for w in pair})
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    means = {}
    for w in windows:
        rolled = np.full(values.size, np.nan)
        rolled[w - 1:] = (cumulative[w:] - cumulative[: cumulative.size - w]) / w
        means[w] = rolled

    warmup = max(s for _, s in params) + 1
    rows = np.empty((len(params), values.size - warmup))
    for i, (f, s) in enumerate(params):
        signal = np.sign(means[f] - means[s])
        position = np.empty_like(signal)
        position[0] = np.nan
        position[1:] = signal[:-1]
        turnover = np.abs(np.diff(position, prepend=position[0]))
        turnover[np.isnan(turnover)] = 0.0
        rows[i] = (position * asset_returns - cost * turnover)[warmup:]

    return rows, params
