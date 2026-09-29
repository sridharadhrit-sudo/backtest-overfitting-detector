"""Backtest Overfitting Detector -- interactive dashboard.

Every number on this page comes from src/. This file only asks for it and draws it.
Run locally with:  uv run streamlit run app.py
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import charts
from src.backtest import fast_sweep
from src.data import load_prices, to_returns
from src.dsr import deflate_family, dsr_curve
from src.effective_trials import bootstrap_family_null, spectral_effective_trials
from src.pbo import cscv
from src.sharpe import min_track_record_length
from src.units import annualise

BUNDLED = Path("sample_data")
MAX_VARIANTS = 3000

st.set_page_config(page_title="Backtest Overfitting Detector", layout="wide")


# ---------------------------------------------------------------- cached work

@st.cache_data(show_spinner="Loading prices...")
def get_prices(ticker, start):
    bundled = BUNDLED / f"{ticker}_{start.replace('-', '')}_latest.csv"
    cache_dir = BUNDLED if bundled.exists() else Path("data")
    return load_prices(ticker, start, cache_dir=str(cache_dir))


@st.cache_data(show_spinner="Backtesting every variant...")
def get_sweep(ticker, start, fast, slow, cost):
    prices = get_prices(ticker, start)
    matrix, params = fast_sweep(prices, fast, slow, cost)
    return matrix, params


@st.cache_data(show_spinner="Running cross-validation over every split...")
def get_cscv(ticker, start, fast, slow, cost, blocks):
    matrix, _ = get_sweep(ticker, start, fast, slow, cost)
    return cscv(matrix, n_blocks=blocks)


@st.cache_data(show_spinner=False)
def get_spectral(ticker, start, fast, slow, cost):
    matrix, _ = get_sweep(ticker, start, fast, slow, cost)
    return spectral_effective_trials(matrix)


# ---------------------------------------------------------------- controls

# The configured theme decides what the page actually renders; ask it first, so the
# charts can never draw in dark-mode colours on a light page (or the reverse).
theme_mode = st.get_option("theme.base") or st.context.theme.type
palette = charts.PALETTES["dark" if theme_mode == "dark" else "light"]

with st.sidebar:
    st.header("The strategy family")
    ticker = st.text_input("Ticker", "SPY").strip().upper()
    start = st.date_input("History starts", date(2008, 1, 1),
                          min_value=date(1995, 1, 1)).isoformat()
    fast_lo, fast_hi = st.slider("Fast window range (days)", 2, 120, (2, 60))
    fast_step = st.select_slider("Fast window step", [1, 2, 5, 10], 2)
    slow_lo, slow_hi = st.slider("Slow window range (days)", 10, 400, (20, 300))
    slow_step = st.select_slider("Slow window step", [5, 10, 20, 50], 10)
    cost_bps = st.number_input("Transaction cost (basis points)", 0.0, 50.0, 1.0, 0.5)

    st.header("The tests")
    blocks = st.select_slider("Cross-validation blocks", [8, 10, 12, 14, 16], 16,
                              help="16 blocks gives 12,870 symmetric splits.")
    n_boot = st.slider("Bootstrap resamples", 50, 500, 150, 50)
    block_len = st.slider("Bootstrap block length (days)", 1, 63, 21,
                          help="About a month keeps volatility clustering intact.")

fast = tuple(range(fast_lo, fast_hi + 1, fast_step))
slow = tuple(range(slow_lo, slow_hi + 1, slow_step))
cost = cost_bps / 10_000
n_pairs = sum(1 for f in fast for s in slow if f < s)
key = (ticker, start, fast, slow, cost)

# ---------------------------------------------------------------- header

st.title("Backtest Overfitting Detector")
st.markdown(
    "Sweep a family of moving-average strategies, keep the best one, then find out "
    "how much of its Sharpe ratio is **skill** and how much is the reward for "
    "**having searched**. Change anything in the sidebar and every test reruns."
)

if n_pairs == 0:
    st.error("No valid pairs: every fast window must be shorter than some slow window.")
    st.stop()
if n_pairs > MAX_VARIANTS:
    st.error(f"That grid has {n_pairs:,} variants. Widen the steps to stay under "
             f"{MAX_VARIANTS:,} so the page stays responsive.")
    st.stop()

try:
    prices = get_prices(ticker, start)
    matrix, params = get_sweep(*key)
except Exception as error:  # a bad ticker, no network, too little history
    st.error(f"Could not build the sweep for {ticker}: {error}")
    st.stop()

if matrix.shape[1] < 2 * blocks * 10:
    st.error("Not enough history after the warm-up period. Start earlier or shorten "
             "the slow windows.")
    st.stop()

n_variants, n_days = matrix.shape
report = deflate_family(matrix)
trial_srs = annualise(matrix.mean(axis=1) / matrix.std(axis=1, ddof=1))
winner_sr = annualise(report["sharpe"])
threshold = annualise(report["threshold"])
fast_w, slow_w = params[report["winner_index"]]
buy_hold = to_returns(prices)
buy_hold_sr = annualise(buy_hold.mean() / buy_hold.std(ddof=1))
cv = get_cscv(*key, blocks)

boot_key = (*key, n_boot, block_len)
boot = st.session_state.get("boot") if st.session_state.get("boot_key") == boot_key else None

# ---------------------------------------------------------------- verdict

boot_p = None if boot is None else float((boot >= report["sharpe"]).mean())
measured = None if boot is None else float(annualise(boot.mean()))

tiles = st.columns(6)
tiles[0].metric("Winner's Sharpe", f"{winner_sr:.2f}", help=f"fast {fast_w}, slow {slow_w}")
tiles[1].metric("PSR vs zero", f"{report['psr_vs_zero']:.1%}",
                help="The number people usually publish.")
tiles[2].metric("Deflated Sharpe", f"{report['dsr']:.3f}",
                help=f"Against the formula's bar for {n_variants:,} trials.")
tiles[3].metric("Overfitting probability", f"{cv['pbo']:.1%}",
                help="How often the in-sample winner lands in the bottom half out of sample.")
tiles[4].metric("Bootstrap P", "—" if boot_p is None else f"{boot_p:.2f}",
                help="Run the measured null below to fill this in.")
tiles[5].metric(f"Buy and hold {ticker}", f"{buy_hold_sr:.2f}")

luck = cv["pbo"] > 0.5 or (boot_p is not None and boot_p > 0.10)
solid = cv["pbo"] < 0.2 and report["dsr"] > 0.95 and (boot_p is None or boot_p < 0.05)
if solid:
    st.success(f"**Survives.** The winner clears the bar and keeps working out of sample. "
               f"Worth taking seriously — then test it on data it has never seen.")
elif luck:
    extra = "" if boot_p is None else f" {boot_p:.0%} of structureless histories beat it."
    st.error(f"**Consistent with luck.** The in-sample winner lands in the bottom half "
             f"out of sample {cv['pbo']:.0%} of the time.{extra}")
else:
    st.warning("**Inconclusive.** The tests disagree, or the margin is thin. "
               "Run the measured null before believing anything.")

st.caption(f"{ticker} · {prices.index[0]:%b %Y} – {prices.index[-1]:%b %Y} · "
           f"{n_variants:,} variants over {n_days:,} common trading days "
           f"({n_days/252:.1f} years) · {cost_bps:g}bp costs · long/short")

# ---------------------------------------------------------------- tabs

tab_bar, tab_dsr, tab_cv, tab_null, tab_data = st.tabs(
    ["The winner vs the bar", "Deflation", "Cross-validation", "Measured null", "Data"])

with tab_bar:
    left, right = st.columns([1, 1])
    with left:
        st.altair_chart(charts.sharpe_histogram(trial_srs, threshold, winner_sr,
                                                palette, measured),
                        use_container_width=True)
        st.caption("Every variant's Sharpe. The dashed line is the best that "
                   f"{n_variants:,} worthless strategies would reach by luck, according "
                   "to the formula. The solid line, once you run the measured null, is "
                   "the same thing measured directly. The orange line is your winner.")
    with right:
        st.altair_chart(charts.grid_heatmap(params, trial_srs, report["winner_index"],
                                            palette),
                        use_container_width=True)
        st.caption("Sharpe across the grid. Broad smooth regions mean neighbouring "
                   "variants are nearly the same strategy — far fewer than "
                   f"{n_variants:,} independent attempts.")

with tab_dsr:
    spectral = get_spectral(*key)
    candidates = sorted({1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
                         10_000, 20_000, n_variants})
    chosen = st.select_slider("Assume this many independent trials", candidates,
                              value=n_variants)
    trials = np.unique(np.round(np.geomspace(2, 20_000, 80)).astype(int))
    curve = dsr_curve(report, n_days, trials)
    chosen_dsr = float(dsr_curve(report, n_days, [chosen])[0])
    marks = {f"raw grid {n_variants:,}": n_variants, f"spectral {spectral:.1f}": spectral}
    st.altair_chart(charts.dsr_curve_chart(trials, curve, marks, chosen, palette),
                    use_container_width=True)
    mintrl = min_track_record_length(report["sharpe"], report["threshold"],
                                     report["skewness"], report["kurtosis"])
    a, b, c = st.columns(3)
    a.metric(f"DSR at {chosen:,} trials", f"{chosen_dsr:.3f}")
    b.metric("Spectral effective trials", f"{spectral:.1f}")
    c.metric("Years of data needed", "∞" if not np.isfinite(mintrl) else f"{mintrl/252:,.0f}")
    st.caption("The deflation needs a count of independent trials, and the verdict moves "
               "a long way depending on what you assume. Structural measures like the "
               "spectral count describe directions in return space, not chances for luck "
               "— trusting them can flip the answer the wrong way. The measured null tab "
               "avoids the question entirely.")

with tab_cv:
    left, right = st.columns(2)
    with left:
        st.altair_chart(charts.logit_histogram(cv["logits"], cv["pbo"], palette),
                        use_container_width=True)
        st.caption(f"{cv['n_splits']:,} symmetric splits of {blocks} blocks. Left of zero "
                   "means the in-sample winner finished in the bottom half out of sample. "
                   "Pure noise puts half the mass there; more than half is the winner's "
                   "curse.")
    with right:
        st.altair_chart(charts.degradation_scatter(
            annualise(cv["is_sharpe"]), annualise(cv["oos_sharpe"]),
            cv["degradation_slope"], annualise(cv["degradation_intercept"]), palette),
            use_container_width=True)
        st.caption(f"Slope {cv['degradation_slope']:+.2f}. Real skill slopes upward: "
                   "looking good in sample would predict doing well out of it.")
    a, b, c = st.columns(3)
    a.metric("Median winner, in sample", f"{annualise(np.median(cv['is_sharpe'])):+.2f}")
    b.metric("Median winner, out of sample", f"{annualise(cv['median_oos_sharpe']):+.2f}")
    c.metric("P(out-of-sample loss)", f"{cv['prob_oos_loss']:.1%}")

with tab_null:
    st.markdown(
        "Resample the real daily returns in blocks, destroying every trend a "
        "moving average could follow, rebuild a price path, and rerun the **entire** "
        "sweep. Repeat. The variants stay exactly as correlated as they really are, so "
        "this measures what *best of my variants, by luck alone* means — no assumption "
        "about independent trials anywhere."
    )
    if st.button(f"Run {n_boot} resamples", type="primary"):
        bar = st.progress(0.0, text="Resampling...")
        result, _ = bootstrap_family_null(
            prices, fast, slow, n_bootstrap=n_boot, block_length=block_len, cost=cost,
            seed=7, on_step=lambda done, total: bar.progress(done / total,
                                                              text=f"{done}/{total}"))
        bar.empty()
        st.session_state["boot"], st.session_state["boot_key"] = result, boot_key
        st.rerun()
    if boot is None:
        st.info("Not run yet for these settings. It takes about ten seconds per hundred "
                "resamples.")
    else:
        st.altair_chart(charts.bootstrap_histogram(annualise(boot), winner_sr, boot_p,
                                                   palette),
                        use_container_width=True)
        a, b, c = st.columns(3)
        a.metric("Measured bar (mean best)", f"{measured:.2f}",
                 delta=f"{measured - threshold:+.2f} vs formula", delta_color="off")
        b.metric("95th percentile", f"{annualise(np.percentile(boot, 95)):.2f}")
        c.metric("P(resampled beats real)", f"{boot_p:.2f}")

with tab_data:
    table = pd.DataFrame(params, columns=["fast", "slow"])
    table["annualised Sharpe"] = trial_srs.round(4)
    table = table.sort_values("annualised Sharpe", ascending=False).reset_index(drop=True)
    table.index += 1
    st.dataframe(table, use_container_width=True, height=420)
    summary = {
        "variants": n_variants, "trading days": n_days,
        "winner (fast, slow)": f"{fast_w}, {slow_w}",
        "winner Sharpe": round(winner_sr, 4),
        "formula threshold": round(threshold, 4),
        "PSR vs zero": round(report["psr_vs_zero"], 4),
        "Deflated Sharpe Ratio": round(report["dsr"], 4),
        "PBO": round(cv["pbo"], 4),
        "degradation slope": round(cv["degradation_slope"], 4),
        "bootstrap P": "not run" if boot_p is None else round(boot_p, 4),
    }
    st.dataframe(pd.DataFrame(summary.items(), columns=["measure", "value"]),
                 hide_index=True, use_container_width=True)
