"""Worked example: what an annualised Sharpe of 1.5 is actually worth."""

import numpy as np

from src.sharpe import (
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_std_error,
)
from src.units import annualise, deannualise

ANNUAL_SR = 1.50
YEARS = 3
N_PERIODS = YEARS * 252
NOISE_THRESHOLD_ANNUAL = 1.46          # E[max SR] from Stage 1, N = 1000

CASES = {
    "normal returns": (0.0, 3.0),
    "skewed, fat-tailed": (-1.2, 8.0),
}


def main():
    sr = deannualise(ANNUAL_SR)
    benchmark = deannualise(NOISE_THRESHOLD_ANNUAL)

    print(f"Annualised Sharpe {ANNUAL_SR}, {YEARS} years of daily data "
          f"(per-period SR = {sr:.4f})\n")

    header = f"{'':28}" + "".join(f"{name:>22}" for name in CASES)
    print(header)

    rows = {}
    for name, (g3, g4) in CASES.items():
        se = sharpe_std_error(sr, N_PERIODS, g3, g4)
        rows[name] = {
            "skewness": g3,
            "kurtosis": g4,
            "SE (per period)": se,
            "SE (annualised)": annualise(se),
            "PSR vs 0": probabilistic_sharpe_ratio(sr, N_PERIODS, 0.0, g3, g4),
            "PSR vs 1.46": probabilistic_sharpe_ratio(sr, N_PERIODS, benchmark, g3, g4),
            "MinTRL vs 0 (yrs)": min_track_record_length(sr, 0.0, g3, g4) / 252,
            "MinTRL vs 1.46 (yrs)": min_track_record_length(sr, benchmark, g3, g4) / 252,
        }

    for key in next(iter(rows.values())):
        line = f"{key:28}"
        for name in CASES:
            value = rows[name][key]
            line += f"{value:>22.4f}" if np.isfinite(value) else f"{'inf':>22}"
        print(line)

    print(f"\n1/sqrt(years) = {1/np.sqrt(YEARS):.4f}  <- compare the annualised SE row")


if __name__ == "__main__":
    main()
