"""Place the case study's DSR inside a null distribution of the SAME shape.

Stage 3 showed the DSR is not uniform, so "0.66" cannot be read against the usual
0.95 convention. The fix is to build the reference distribution the result actually
needs: families of exactly the same size, over exactly the same number of days, in
which nothing has any edge. Then ask what fraction of those reach the observed value.

This is a permutation-style test, and it only exists because Stage 1 built a null world.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.dsr import deflate_family
from src.nulls import simulate_null_returns

N_EXPERIMENTS = 200   # raise for a tighter p-value; runtime scales linearly
SEED = 2026


def main():
    matrix = np.load("data/sweep_returns.npy")
    n_trials, n_periods = matrix.shape
    observed = deflate_family(matrix)["dsr"]

    print(f"case study : {n_trials} variants over {n_periods} days "
          f"({n_periods/252:.1f} years)")
    print(f"observed DSR : {observed:.4f}")
    print(f"simulating {N_EXPERIMENTS} no-edge families of the same shape...")

    rng = np.random.default_rng(SEED)
    null = np.empty(N_EXPERIMENTS)
    for i in range(N_EXPERIMENTS):
        null[i] = deflate_family(simulate_null_returns(n_trials, n_periods, rng))["dsr"]

    percentile = (null < observed).mean()
    p_value = 1 - percentile

    print()
    print(f"null mean DSR            : {null.mean():.4f}")
    print(f"null 5th / 50th / 95th   : {np.percentile(null, 5):.3f} / "
          f"{np.percentile(null, 50):.3f} / {np.percentile(null, 95):.3f}")
    print(f"observed sits at the     : {100*percentile:.1f}th percentile")
    print(f"P(no-edge family >= obs) : {p_value:.3f}")
    print()
    if p_value > 0.10:
        print("VERDICT: comfortably inside what pure noise produces. Not evidence.")
    elif p_value > 0.05:
        print("VERDICT: marginal. Suggestive at best, and not by conventional standards.")
    else:
        print("VERDICT: beyond what this much searching produces by luck. Worth pursuing.")

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(null, bins=28, alpha=0.85, label=f"{N_EXPERIMENTS} no-edge families")
    ax.axvline(observed, color="crimson", lw=1.8,
               label=f"observed DSR {observed:.3f}")
    ax.axvline(np.percentile(null, 95), color="black", lw=1.3, ls="--",
               label=f"null 95th pct {np.percentile(null, 95):.3f}")
    ax.set_xlabel("Deflated Sharpe Ratio")
    ax.set_ylabel("families")
    ax.set_title(f"Where the result sits among {n_trials} trials of nothing")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("figures/null_reference.png", dpi=160)
    print("\nsaved figures/null_reference.png")


if __name__ == "__main__":
    main()
