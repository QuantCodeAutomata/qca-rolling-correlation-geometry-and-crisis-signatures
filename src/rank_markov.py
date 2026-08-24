"""Rank-space Markov chain construction and diagnostics.

Implements exp_5: build decile state sequences from rolling return and
rolling volatility ranks, estimate transition matrices, compute mixing
times, migration rates, sector coherence, and an independent-return
null baseline.

# Custom — Context7 confirmed direct-counting transition estimation is
# more faithful for observable chains than hmmlearn/pomegranate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg


DEFAULT_DECILES = 10


def rolling_mean_return(returns: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Trailing rolling mean daily return per stock."""
    return returns.rolling(window=window, min_periods=window).mean()


def rolling_vol(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Trailing rolling sample standard deviation per stock."""
    return returns.rolling(window=window, min_periods=window).std(ddof=1)


def cross_sectional_decile(stat: pd.DataFrame, n_deciles: int = DEFAULT_DECILES) -> pd.DataFrame:
    """Convert cross-sectional statistic into balanced decile classes.

    For each date, tickers are ranked (with 'first' tie-break) and mapped
    to deciles 0..n_deciles-1 using a balanced allocation. Class sizes
    differ by at most one.
    """
    def _row_to_decile(row: pd.Series) -> pd.Series:
        s = row.dropna().sort_values(kind="mergesort")
        n = len(s)
        if n < n_deciles:
            return pd.Series(index=row.index, dtype="float64")
        idx_positions = np.arange(n)
        # balanced allocation
        boundaries = np.linspace(0, n, n_deciles + 1).astype(int)
        deciles = np.zeros(n, dtype=int)
        for d in range(n_deciles):
            deciles[boundaries[d] : boundaries[d + 1]] = d
        out = pd.Series(deciles, index=s.index, dtype=float)
        return out.reindex(row.index)

    return stat.apply(_row_to_decile, axis=1)


def transition_matrix(states: pd.DataFrame, n_states: int) -> np.ndarray:
    """Estimate a row-stochastic transition matrix from wide state series.

    Pools transitions across stocks and dates. NaN transitions are
    skipped. Rows with zero counts are set to uniform to preserve row-
    stochasticity, and this convention is documented.
    """
    a = states.values
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(a.shape[1]):
        col = a[:, i]
        for t in range(len(col) - 1):
            s = col[t]
            s2 = col[t + 1]
            if np.isnan(s) or np.isnan(s2):
                continue
            counts[int(s), int(s2)] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(row_sums > 0, counts / row_sums, 1.0 / n_states)
    return p


def stay_probabilities(p: np.ndarray) -> np.ndarray:
    """Diagonal entries of a transition matrix."""
    return np.diag(p).copy()


def second_eigenvalue_modulus(p: np.ndarray) -> float:
    """|lambda_2| of a row-stochastic matrix."""
    vals = linalg.eigvals(p)
    mag = np.abs(vals)
    order = np.argsort(mag)[::-1]
    return float(mag[order[1]])


def mixing_time(p: np.ndarray) -> float:
    """tau_mix = -1 / log|lambda_2|."""
    l2 = second_eigenvalue_modulus(p)
    if l2 <= 0 or l2 >= 1:
        return float("inf")
    return float(-1.0 / np.log(l2))


def migration_rate(states: pd.DataFrame, smooth: int = 21) -> pd.Series:
    """Mean absolute one-step class change per stock, optionally smoothed."""
    diff = states.diff().abs()
    mean_daily = diff.mean(axis=1)
    if smooth and smooth > 1:
        mean_daily = mean_daily.rolling(smooth, min_periods=1).mean()
    return mean_daily.rename("migration")


def rolling_transition_diagnostics(
    states: pd.DataFrame, window: int, n_states: int, step: int = 5
) -> pd.DataFrame:
    """Compute rolling |lambda_2|, mixing time, and mean stay probability."""
    dates = states.index
    rows = []
    for end in range(window - 1, len(dates), step):
        start = end - window + 1
        chunk = states.iloc[start : end + 1]
        p = transition_matrix(chunk, n_states=n_states)
        rows.append(
            {
                "date": dates[end],
                "lambda2": second_eigenvalue_modulus(p),
                "tau_mix": mixing_time(p),
                "mean_stay": float(np.diag(p).mean()),
            }
        )
    return pd.DataFrame(rows)


def sector_coherence(
    states: pd.DataFrame, sectors: pd.Series
) -> tuple[float, float]:
    """Return (same_sector_corr, all_pair_corr) of one-step rank changes.

    Correlations are averaged pairwise Pearson correlations of the
    delta-state series across stocks.
    """
    deltas = states.diff().dropna(how="any")
    tickers = deltas.columns
    corr = deltas.corr()
    n_ticks = len(tickers)
    same_pairs = []
    all_pairs = []
    sec = sectors.reindex(tickers)
    for i in range(n_ticks):
        for j in range(i + 1, n_ticks):
            c = corr.iloc[i, j]
            if np.isnan(c):
                continue
            all_pairs.append(c)
            if sec.iloc[i] == sec.iloc[j]:
                same_pairs.append(c)
    same_avg = float(np.mean(same_pairs)) if same_pairs else float("nan")
    all_avg = float(np.mean(all_pairs)) if all_pairs else float("nan")
    return same_avg, all_avg


def independent_return_null(
    returns: pd.DataFrame,
    stat_fn,
    n_states: int = DEFAULT_DECILES,
    seed: int = 0,
) -> np.ndarray:
    """Independent-return null: shuffle each column's returns in time.

    ``stat_fn`` takes the reshuffled returns frame and returns a wide
    frame of the ranking statistic; this function then coarse-grains it
    and estimates the pooled transition matrix.
    """
    rng = np.random.default_rng(seed)
    shuffled = returns.apply(lambda col: pd.Series(rng.permutation(col.values), index=col.index), axis=0)
    stat = stat_fn(shuffled)
    deciles = cross_sectional_decile(stat, n_deciles=n_states)
    return transition_matrix(deciles, n_states=n_states)
