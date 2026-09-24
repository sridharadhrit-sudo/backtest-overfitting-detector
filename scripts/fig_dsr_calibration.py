"""Calibration: does the DSR behave the way a probability should under the null?

Run the detector on families where NOTHING has any edge, and look at the distribution
of the answers. Two things matter: the centre (is it 0.5?) and the tails (how often
does it wrongly bless noise?).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

from src.dsr import deflate_family
from src.nulls import simulate_null_returns
from src.sharpe import sharpe_std_error

N_STRATEGIES = 100
N_PERIODS = 5 * 252
N_EXPERIMENTS = 1000
SEED = 11


def main():
    rng = np.random.default_rng(SEED)
    dsr = np.empty(N_EXPERIMENTS)
    psr = np.empty(N_EXPERIMENTS)
    winner_sr = np.empty(N_EXPERIMENTS)

    for i in range(N_EXPERIMENTS):
        family = simulate_null_returns(N_STRATEGIES, N_PERIODS, rng)
        report = deflate_family(family)
        dsr[i] = report["dsr"]
        psr[i] = report["psr_vs_zero"]
        winner_sr[i] = report["sharpe"]

    ks = kstest(dsr, "uniform")
    se_used = sharpe_std_error(winner_sr.mean(), N_PERIODS)
    spread_of_max = winner_sr.std(ddof=1)

    print(f"experiments                    : {N_EXPERIMENTS}")
    print()
    print("CENTRE -- the deflation is unbiased")
    print(f"  mean DSR                     : {dsr.mean():.4f}    (want 0.50)")
    print(f"  mean PSR vs zero             : {psr.mean():.4f}    (undeflated, useless)")
    print()
    print("TAILS -- false positives under the null")
    print(f"  DSR  > 0.95                  : {(dsr > 0.95).mean():.4f}")
    print(f"  PSR  > 0.95                  : {(psr > 0.95).mean():.4f}")
    print()
    print("SHAPE -- is DSR a p-value?")
    print(f"  KS statistic vs uniform      : {ks.statistic:.4f}  (p = {ks.pvalue:.3g})")
    print(f"  DSR 5th / 95th percentile    : {np.percentile(dsr, 5):.3f} / "
          f"{np.percentile(dsr, 95):.3f}   (uniform -> 0.05 / 0.95)")
    print()
    print("WHY -- the denominator is the wrong yardstick")
    print(f"  std error of one Sharpe      : {se_used:.5f}   <- what DSR divides by")
    print(f"  actual spread of the MAXIMUM : {spread_of_max:.5f}")
    print(f"  ratio                        : {se_used / spread_of_max:.2f}x too wide")

    grid = np.linspace(0, 1, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.hist(dsr, bins=25, range=(0, 1), density=True, alpha=0.85, label="DSR")
    ax1.axhline(1.0, color="black", lw=1.4, ls="--", label="uniform density")
    ax1.set_xlabel("DSR under the null")
    ax1.set_ylabel("density")
    ax1.set_title("Centred on 0.5, but too narrow")
    ax1.legend(frameon=False)

    ax2.plot(grid, np.searchsorted(np.sort(dsr), grid) / dsr.size, lw=2, label="DSR")
    ax2.plot(grid, np.searchsorted(np.sort(psr), grid) / psr.size, lw=2,
             label="PSR vs zero (undeflated)")
    ax2.plot(grid, grid, color="black", lw=1.2, ls="--", label="uniform")
    ax2.set_xlabel("value")
    ax2.set_ylabel("empirical CDF")
    ax2.set_title("What deflation fixes, and what it does not")
    ax2.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig("figures/dsr_calibration.png", dpi=160)
    print("\nsaved figures/dsr_calibration.png")


if __name__ == "__main__":
    main()
