"""The Sharpe ratio as an estimator: standard error, PSR, minimum track record length.

Bailey & Lopez de Prado (2012, 2014); the non-normal standard error is Mertens (2002),
which generalises Lo (2002).

Every Sharpe ratio in this module is PER PERIOD. Use src.units to convert.
"""

import numpy as np
from scipy.stats import kurtosis, norm, skew


def sharpe_ratio(returns, risk_free=0.0):
    """Per-period Sharpe ratio of a single return series."""
    excess = np.asarray(returns, dtype=float) - risk_free
    return excess.mean() / excess.std(ddof=1)


def moments(returns):
    """Sample skewness (gamma_3) and kurtosis (gamma_4).

    Kurtosis is NOT excess: gamma_4 = 3.0 for a normal distribution.
    """
    r = np.asarray(returns, dtype=float)
    return skew(r, bias=False), kurtosis(r, fisher=False, bias=False)


def sharpe_variance_factor(sr, gamma_3=0.0, gamma_4=3.0):
    """The bracket 1 - gamma_3*SR + (gamma_4-1)/4 * SR**2.

    Equals 1 + SR**2/2 for normal returns. Appears in both PSR and MinTRL.
    """
    return 1.0 - gamma_3 * sr + (gamma_4 - 1.0) / 4.0 * sr ** 2


def sharpe_std_error(sr, n_periods, gamma_3=0.0, gamma_4=3.0):
    """Standard error of the per-period Sharpe estimator."""
    return np.sqrt(sharpe_variance_factor(sr, gamma_3, gamma_4) / (n_periods - 1))


def probabilistic_sharpe_ratio(sr, n_periods, benchmark=0.0, gamma_3=0.0, gamma_4=3.0):
    """P(true SR > benchmark), given sample length and the first four moments."""
    return norm.cdf((sr - benchmark) / sharpe_std_error(sr, n_periods, gamma_3, gamma_4))


def min_track_record_length(sr, benchmark=0.0, gamma_3=0.0, gamma_4=3.0, confidence=0.95):
    """Observations needed before PSR(benchmark) would reach `confidence`.

    Infinite if the observed Sharpe does not exceed the benchmark: no amount of
    further data makes a losing comparison significant.
    """
    if sr <= benchmark:
        return np.inf
    z = norm.ppf(confidence)
    return 1.0 + sharpe_variance_factor(sr, gamma_3, gamma_4) * (z / (sr - benchmark)) ** 2


def sharpe_report(returns, benchmark=0.0, confidence=0.95, risk_free=0.0):
    """Everything above, computed from one return series."""
    r = np.asarray(returns, dtype=float)
    sr = sharpe_ratio(r, risk_free)
    g3, g4 = moments(r)
    n = r.size
    return {
        "sharpe": sr,
        "skewness": g3,
        "kurtosis": g4,
        "n_periods": n,
        "std_error": sharpe_std_error(sr, n, g3, g4),
        "psr": probabilistic_sharpe_ratio(sr, n, benchmark, g3, g4),
        "min_track_record": min_track_record_length(sr, benchmark, g3, g4, confidence),
    }
