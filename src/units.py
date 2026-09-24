"""Time-unit conversions. One place, so the sqrt(252) can only be wrong once."""

import numpy as np

TRADING_DAYS = 252


def annualise(sharpe_per_period, periods_per_year=TRADING_DAYS):
    """Per-period Sharpe -> annualised Sharpe."""
    return sharpe_per_period * np.sqrt(periods_per_year)


def deannualise(sharpe_annual, periods_per_year=TRADING_DAYS):
    """Annualised Sharpe -> per-period Sharpe. The DSR formulas want this one."""
    return sharpe_annual / np.sqrt(periods_per_year)
