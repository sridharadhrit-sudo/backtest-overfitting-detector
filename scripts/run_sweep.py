"""Stage 4: sweep a strategy family over real prices, then put the winner on trial."""

import matplotlib.pyplot as plt
import numpy as np

from src.backtest import DEFAULT_COST, sweep
from src.data import load_prices, to_returns
from src.dsr import deflate_family
from src.nulls import sharpe_ratios
from src.sharpe import min_track_record_length
from src.units import annualise

TICKER = "SPY"
START = "2008-01-01"
FAST_WINDOWS = range(2, 62, 2)      # 30 values
SLOW_WINDOWS = range(20, 310, 10)   # 29 values


def main():
    prices = load_prices(TICKER, START)
    matrix, params = sweep(prices, FAST_WINDOWS, SLOW_WINDOWS, cost=DEFAULT_COST)
    n_variants, n_days = matrix.shape
    years = n_days / 252

    report = deflate_family(matrix)
    fast, slow = params[report["winner_index"]]
    all_sr = annualise(sharpe_ratios(matrix))

    buy_hold = to_returns(prices)
    bh_sr = annualise(buy_hold.mean() / buy_hold.std(ddof=1))

    mintrl = min_track_record_length(
        report["sharpe"], report["threshold"], report["skewness"], report["kurtosis"]
    )

    print(f"{TICKER} from {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"variants swept                 : {n_variants}")
    print(f"common trading days            : {n_days}  ({years:.1f} years)")
    print()
    print(f"best variant                   : fast={fast}, slow={slow}")
    print(f"  annualised Sharpe            : {annualise(report['sharpe']):.4f}")
    print(f"  skewness / kurtosis          : {report['skewness']:.3f} / {report['kurtosis']:.3f}")
    print(f"  PSR vs zero (the usual claim): {report['psr_vs_zero']:.4f}")
    print()
    print(f"noise threshold for {n_variants} trials : {annualise(report['threshold']):.4f}")
    print(f"  DEFLATED SHARPE RATIO        : {report['dsr']:.4f}")
    print(f"  MinTRL vs that threshold     : {mintrl/252:,.0f} years")
    print()
    print(f"median variant Sharpe          : {np.median(all_sr):.4f}")
    print(f"buy and hold {TICKER}              : {bh_sr:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    grid = np.full((len(FAST_WINDOWS), len(SLOW_WINDOWS)), np.nan)
    fast_index = {f: i for i, f in enumerate(FAST_WINDOWS)}
    slow_index = {s: j for j, s in enumerate(SLOW_WINDOWS)}
    for (f, s), sr in zip(params, all_sr):
        grid[fast_index[f], slow_index[s]] = sr
    im = ax1.imshow(grid, aspect="auto", origin="lower", cmap="RdYlBu_r",
                    extent=[min(SLOW_WINDOWS), max(SLOW_WINDOWS),
                            min(FAST_WINDOWS), max(FAST_WINDOWS)])
    ax1.scatter([slow], [fast], marker="o", s=70, facecolors="none",
                edgecolors="black", linewidths=1.6, label="winner")
    ax1.set_xlabel("slow window (days)")
    ax1.set_ylabel("fast window (days)")
    ax1.set_title("Annualised Sharpe across the grid")
    ax1.legend(frameon=False, loc="lower right")
    fig.colorbar(im, ax=ax1)

    ax2.hist(all_sr, bins=40, alpha=0.85, label=f"{n_variants} variants")
    ax2.axvline(annualise(report["threshold"]), color="black", lw=1.6, ls="--",
                label=f"noise threshold {annualise(report['threshold']):.2f}")
    ax2.axvline(annualise(report["sharpe"]), color="crimson", lw=1.6,
                label=f"winner {annualise(report['sharpe']):.2f}")
    ax2.set_xlabel("annualised Sharpe ratio")
    ax2.set_ylabel("variants")
    ax2.set_title("The winner against the bar it has to clear")
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig("figures/sweep.png", dpi=160)
    np.save("data/sweep_returns.npy", matrix)
    print("\nsaved figures/sweep.png and data/sweep_returns.npy")


if __name__ == "__main__":
    main()
