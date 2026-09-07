# Backtest Overfitting Detector

Backtest a thousand trading rules with no real edge, keep the best one, and it will
look excellent. You haven't found a strategy — you've found the right tail of a
sampling distribution. The best Sharpe ratio you find measures how many things you
tried, not how good the strategy is.

This repo builds the tool that quantifies that. Given a family of backtested
strategies, it reports the Deflated Sharpe Ratio, the Probability of Backtest
Overfitting, and how much more data you would need before the result could be
believed at all.

Based on Bailey & Lopez de Prado (2014), *The Deflated Sharpe Ratio*, and
Bailey, Borwein, Lopez de Prado & Zhu (2015), *The Probability of Backtest Overfitting*.

## Status

In progress. Stage 0 of 8: project skeleton.
