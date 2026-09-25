"""Price data: download once, cache to CSV, never hit the network in tests."""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKER = "SPY"
DEFAULT_START = "2008-01-01"


def _cache_path(ticker, start, end, cache_dir):
    tag = f"{ticker}_{start}_{end or 'latest'}".replace("-", "")
    return Path(cache_dir) / f"{tag}.csv"


def load_prices(ticker=DEFAULT_TICKER, start=DEFAULT_START, end=None, cache_dir="data"):
    """Adjusted daily closing prices as a pandas Series indexed by date.

    Downloads from Yahoo on first call and caches the result, so reruns are offline
    and reproducible. Delete the CSV in data/ to refresh.
    """
    path = _cache_path(ticker, start, end, cache_dir)
    if path.exists():
        series = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
        series.name = ticker
        return series

    import yfinance as yf  # imported lazily so tests never need it

    frame = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if frame.empty:
        raise RuntimeError(f"no data returned for {ticker}; check the ticker and dates")

    series = frame["Close"].astype(float)
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series.name = ticker

    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path)
    return series


def to_returns(prices):
    """Simple daily returns, with the first (undefined) observation dropped."""
    return prices.pct_change().dropna()


def synthetic_prices(n_days=3000, annual_drift=0.06, annual_vol=0.18, seed=0):
    """A price series with no exploitable structure. For tests, not for results."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal(
        (annual_drift - 0.5 * annual_vol ** 2) * dt,
        annual_vol * np.sqrt(dt),
        size=n_days,
    )
    index = pd.bdate_range("2005-01-03", periods=n_days)
    return pd.Series(100 * np.exp(np.cumsum(shocks)), index=index, name="SYNTH")
