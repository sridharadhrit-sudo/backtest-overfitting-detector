"""Stage 5: the Probability of Backtest Overfitting for the SPY case study.

A second line of attack on the same question. The DSR is parametric -- it models the
uncertainty in one Sharpe ratio. This makes no distributional assumption at all: it
simply asks whether the selection procedure survives being tested on data it did not
select on.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.pbo import DEFAULT_BLOCKS, cscv
from src.units import annualise

MATRIX_PATH = "data/sweep_returns.npy"


def main():
    matrix = np.load(MATRIX_PATH)
    result = cscv(matrix, n_blocks=DEFAULT_BLOCKS)

    is_sr = annualise(result["is_sharpe"])
    oos_sr = annualise(result["oos_sharpe"])

    print(f"strategies                  : {result['n_strategies']}")
    print(f"blocks                      : {result['n_blocks']} "
          f"of {result['block_length']} days")
    print(f"symmetric splits            : {result['n_splits']:,}")
    print()
    print(f"PROBABILITY OF OVERFITTING  : {result['pbo']:.4f}")
    print(f"  the in-sample winner lands in the bottom half out of sample")
    print(f"  in {100*result['pbo']:.1f}% of splits")
    print()
    print(f"median winner, in sample    : {np.median(is_sr):+.4f} annualised")
    print(f"median winner, out of sample: {np.median(oos_sr):+.4f} annualised")
    print(f"P(out-of-sample loss)       : {result['prob_oos_loss']:.4f}")
    print(f"degradation slope           : {result['degradation_slope']:+.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax1.hist(result["logits"], bins=40, alpha=0.85)
    ax1.axvline(0, color="black", lw=1.6, ls="--", label="upper half / lower half")
    ax1.set_xlabel("logit of the winner's out-of-sample rank")
    ax1.set_ylabel("splits")
    ax1.set_title(f"PBO = {result['pbo']:.3f}: mass sits left of zero")
    ax1.legend(frameon=False)

    ax2.scatter(is_sr, oos_sr, s=4, alpha=0.25)
    line = np.linspace(is_sr.min(), is_sr.max(), 50)
    slope = result["degradation_slope"]
    intercept = result["degradation_intercept"]
    ax2.plot(line, annualise(slope * (line / np.sqrt(252)) + intercept),
             color="crimson", lw=1.8, label=f"fit, slope {slope:+.2f}")
    ax2.axhline(0, color="black", lw=1.0)
    ax2.set_xlabel("winner's in-sample Sharpe (annualised)")
    ax2.set_ylabel("same winner, out of sample")
    ax2.set_title("Performance degradation")
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig("figures/pbo.png", dpi=160)
    print("\nsaved figures/pbo.png")


if __name__ == "__main__":
    main()
