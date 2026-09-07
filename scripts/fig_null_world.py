"""Figure: the distribution of the best Sharpe when nothing has any edge."""

import matplotlib.pyplot as plt
import numpy as np

from src.nulls import (
    annualise,
    expected_max_sharpe,
    max_sharpe_distribution,
    sharpe_ratios,
    simulate_null_returns,
)

N_STRATEGIES = 1000
N_PERIODS = 5 * 252          # five years of daily data
N_EXPERIMENTS = 200


def main():
    rng = np.random.default_rng(0)

    # One representative experiment: the cross-section of 1,000 worthless strategies.
    one_run = annualise(sharpe_ratios(simulate_null_returns(N_STRATEGIES, N_PERIODS, rng)))

    # Many experiments: how the winner is distributed.
    winners = annualise(
        max_sharpe_distribution(N_STRATEGIES, N_PERIODS, N_EXPERIMENTS, seed=1)
    )

    # Analytic prediction. sigma_SR = 1/sqrt(T) per period, so 1/sqrt(years) annualised.
    sigma_sr_annual = 1 / np.sqrt(N_PERIODS / 252)
    predicted = expected_max_sharpe(N_STRATEGIES, sigma_sr_annual)

    print(f"simulated mean best Sharpe : {winners.mean():.4f}")
    print(f"analytic E[max SR]         : {predicted:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(one_run, bins=50, alpha=0.55, label="all 1,000 strategies (no edge)")
    ax.hist(winners, bins=30, alpha=0.85, label=f"the winner, over {N_EXPERIMENTS} runs")
    ax.axvline(predicted, color="black", lw=1.6, ls="--",
               label=f"analytic E[max SR] = {predicted:.2f}")
    ax.set_xlabel("annualised Sharpe ratio")
    ax.set_ylabel("count")
    ax.set_title("Nothing here has any edge")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("figures/null_max_sharpe.png", dpi=160)
    print("saved figures/null_max_sharpe.png")


if __name__ == "__main__":
    main()
