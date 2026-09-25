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
