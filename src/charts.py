"""Chart specifications for the dashboard.

Pure presentation: every function takes numbers the library has already computed and
returns an Altair chart. Nothing here calculates anything that matters.

Colour follows one rule set, in both themes:
  series  -- the distribution being shown (slot 1, blue)
  accent  -- the one value that matters: the winner, the fit line (slot 2, orange)
  ink     -- thresholds and reference lines, in the text colour, never a series colour
  heatmap -- diverging red/grey/blue around zero, because a Sharpe ratio has a sign
"""

import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.disable_max_rows()

PALETTES = {
    "light": {"series": "#2a78d6", "accent": "#eb6834", "ink": "#0b0b0b",
              "muted": "#898781", "neg": "#e34948", "mid": "#f0efec", "pos": "#2a78d6"},
    "dark": {"series": "#3987e5", "accent": "#d95926", "ink": "#ffffff",
             "muted": "#898781", "neg": "#e66767", "mid": "#383835", "pos": "#3987e5"},
}

HEIGHT = 300


def _bars(frame, field, title, colour, bins=40):
    return (
        alt.Chart(frame)
        .mark_bar(color=colour, cornerRadiusTopLeft=2, cornerRadiusTopRight=2,
                  binSpacing=2)
        .encode(
            x=alt.X(f"{field}:Q", bin=alt.Bin(maxbins=bins), title=title),
            y=alt.Y("count():Q", title="count"),
            tooltip=[alt.Tooltip(f"{field}:Q", bin=alt.Bin(maxbins=bins), title=title,
                                 format=".2f"),
                     alt.Tooltip("count():Q", title="count")],
        )
    )


def _rule(value, label, colour, width=2, dash=None, row=0, align="left"):
    """A vertical reference line with its label. `row` staggers labels so neighbouring
    rules never collide; `align="right"` hangs the label to the left of the line."""
    frame = pd.DataFrame({"x": [value], "label": [label]})
    rule = alt.Chart(frame).mark_rule(color=colour, strokeWidth=width,
                                      strokeDash=dash or [])
    rule = rule.encode(x="x:Q", tooltip=[alt.Tooltip("label:N", title=""),
                                         alt.Tooltip("x:Q", title="value", format=".3f")])
    text = alt.Chart(frame).mark_text(align=align, dx=5 if align == "left" else -5,
                                      color=colour, fontSize=12, fontWeight=500)
    text = text.encode(x="x:Q", y=alt.value(10 + 16 * row), text="label:N")
    return rule + text


def sharpe_histogram(sharpes, threshold, winner, palette, measured=None):
    """Every variant's Sharpe, the bar the winner has to clear, and the winner."""
    frame = pd.DataFrame({"sharpe": sharpes})
    layers = [_bars(frame, "sharpe", "annualised Sharpe ratio", palette["series"]),
              _rule(threshold, f"formula bar {threshold:.2f}", palette["ink"], 1.5, [4, 3],
                    row=0, align="right")]
    if measured is not None:
        layers.append(_rule(measured, f"measured bar {measured:.2f}", palette["ink"], 2,
                            row=2))
    layers.append(_rule(winner, f"winner {winner:.2f}", palette["accent"], 2.5, row=1))
    return alt.layer(*layers).properties(height=HEIGHT)


def grid_heatmap(params, sharpes, winner_index, palette):
    """Sharpe across the parameter grid. Smoothness here means correlated variants."""
    frame = pd.DataFrame(params, columns=["fast", "slow"])
    frame["sharpe"] = sharpes
    bound = float(np.nanmax(np.abs(sharpes))) or 1.0
    cells = alt.Chart(frame).mark_rect().encode(
        x=alt.X("slow:O", title="slow window (days)",
                axis=alt.Axis(labelOverlap=True, labelAngle=0)),
        y=alt.Y("fast:O", title="fast window (days)", sort="descending",
                axis=alt.Axis(labelOverlap=True)),
        color=alt.Color("sharpe:Q", title="Sharpe",
                        scale=alt.Scale(domain=[-bound, 0, bound], interpolate="lab",
                                        range=[palette["neg"], palette["mid"],
                                               palette["pos"]])),
        tooltip=["fast:O", "slow:O", alt.Tooltip("sharpe:Q", format=".3f")],
    )
    best = frame.iloc[[winner_index]]
    ring = alt.Chart(best).mark_point(shape="square", size=160, filled=False,
                                      strokeWidth=2.5, color=palette["accent"]).encode(
        x="slow:O", y=alt.Y("fast:O", sort="descending"),
        tooltip=[alt.Tooltip("fast:O", title="winner fast"),
                 alt.Tooltip("slow:O", title="winner slow"),
                 alt.Tooltip("sharpe:Q", format=".3f")],
    )
    return (cells + ring).properties(height=HEIGHT + 60)


def dsr_curve_chart(trials, dsr, marks, chosen, palette):
    """How the verdict moves with the trial count you assume."""
    frame = pd.DataFrame({"trials": trials, "dsr": dsr})
    line = alt.Chart(frame).mark_line(color=palette["series"], strokeWidth=2).encode(
        x=alt.X("trials:Q", scale=alt.Scale(type="log"),
                title="assumed number of independent trials"),
        y=alt.Y("dsr:Q", title="Deflated Sharpe Ratio", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("trials:Q", format=",.0f"),
                 alt.Tooltip("dsr:Q", format=".3f")],
    )
    layers = [line]
    for row, (label, value) in enumerate(marks.items(), start=1):
        layers.append(_rule(value, label, palette["muted"], 1.5, [4, 3], row=row))
    layers.append(_rule(chosen, f"your choice: {chosen:,}", palette["accent"], 2.5, row=0))
    return alt.layer(*layers).properties(height=HEIGHT)


def logit_histogram(logits, pbo, palette):
    frame = pd.DataFrame({"logit": logits})
    return alt.layer(
        _bars(frame, "logit", "logit of the winner's out-of-sample rank",
              palette["series"]),
        _rule(0.0, f"{pbo:.0%} of splits fall left of here", palette["ink"], 2,
              align="right"),
    ).properties(height=HEIGHT)


def degradation_scatter(is_sharpe, oos_sharpe, slope, intercept, palette):
    """Each split's winner: how it looked in sample against how it did out of sample."""
    frame = pd.DataFrame({"in_sample": is_sharpe, "out_of_sample": oos_sharpe})
    dots = alt.Chart(frame).mark_circle(size=12, opacity=0.25,
                                        color=palette["series"]).encode(
        x=alt.X("in_sample:Q", title="winner's Sharpe, in sample",
                scale=alt.Scale(zero=False)),
        y=alt.Y("out_of_sample:Q", title="same winner, out of sample"),
        tooltip=[alt.Tooltip("in_sample:Q", format=".2f"),
                 alt.Tooltip("out_of_sample:Q", format=".2f")],
    )
    lo, hi = float(np.min(is_sharpe)), float(np.max(is_sharpe))
    fit = pd.DataFrame({"in_sample": [lo, hi],
                        "out_of_sample": [slope * lo + intercept, slope * hi + intercept]})
    line = alt.Chart(fit).mark_line(color=palette["accent"], strokeWidth=2.5).encode(
        x="in_sample:Q", y="out_of_sample:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
        color=palette["muted"], strokeWidth=1).encode(y="y:Q")
    return (dots + zero + line).properties(height=HEIGHT)


def bootstrap_histogram(best, observed, p_value, palette):
    frame = pd.DataFrame({"best": best})
    return alt.layer(
        _bars(frame, "best", "best variant's Sharpe on a resampled history",
              palette["series"], bins=30),
        _rule(observed, f"real winner {observed:.2f}  ·  P = {p_value:.2f}",
              palette["accent"], 2.5),
    ).properties(height=HEIGHT)
