"""Entropy production and time-irreversibility in rank chains.

Implements exp_6: joint-probability entropy production, probability
currents, reversible surrogates, rolling z-scores, and firm-level
cumulative entropy production.

# Custom — Context7 confirmed pyinform provides transfer entropy but not
# the paper's exact joint-distribution entropy-production formula.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def joint_transition_probs(
    states: pd.DataFrame, lag: int, n_states: int, pseudo: float = 1e-8
) -> np.ndarray:
    """Estimate the joint probability p(a at t, b at t+lag) pooled across
    stocks and dates.

    Applies a tiny pseudocount ``pseudo`` uniformly to avoid divergences
    in log-ratios; the same convention is applied to observed and
    surrogate calculations (see :func:`reversible_surrogate_pvalue`).
    """
    a = states.values
    counts = np.full((n_states, n_states), pseudo, dtype=np.float64)
    n_cols = a.shape[1]
    for i in range(n_cols):
        col = a[:, i]
        for t in range(len(col) - lag):
            s = col[t]
            s2 = col[t + lag]
            if np.isnan(s) or np.isnan(s2):
                continue
            counts[int(s), int(s2)] += 1.0
    total = counts.sum()
    return counts / total


def entropy_production(mu: np.ndarray) -> float:
    """sigma = sum_ab mu_ab log(mu_ab / mu_ba). Zero-mass terms omitted."""
    n = mu.shape[0]
    sig = 0.0
    for a in range(n):
        for b in range(n):
            mab = mu[a, b]
            mba = mu[b, a]
            if mab > 0.0 and mba > 0.0:
                sig += mab * np.log(mab / mba)
    return float(sig)


def probability_currents(mu: np.ndarray) -> np.ndarray:
    """J_ab = mu_ab - mu_ba."""
    return mu - mu.T


def reversible_surrogate_pvalue(
    mu: np.ndarray, sigma_obs: float, n_draws: int = 200, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean_surr, std_surr, z_score) for surrogate reversibility test.

    Constructs a reversible target mu_rev = (mu + mu.T)/2 and samples
    surrogate joint tables from a multinomial with the same total mass
    to preserve sample size. This is the documented surrogate mechanism
    used throughout the project.
    """
    rng = np.random.default_rng(seed)
    mu_rev = 0.5 * (mu + mu.T)
    total = mu.sum()
    n = mu.shape[0]
    p_flat = (mu_rev / mu_rev.sum()).ravel()
    n_events = int(round(total * 1000.0))  # scale up to reduce variance
    n_events = max(n_events, 500)
    surrogate_sigmas = np.empty(n_draws)
    for k in range(n_draws):
        draw = rng.multinomial(n_events, p_flat).reshape(n, n).astype(np.float64)
        draw = draw / draw.sum()
        surrogate_sigmas[k] = entropy_production(draw)
    mean_s = float(surrogate_sigmas.mean())
    std_s = float(surrogate_sigmas.std(ddof=1) + 1e-12)
    z = float((sigma_obs - mean_s) / std_s)
    return mean_s, std_s, z


def rolling_entropy_production(
    states: pd.DataFrame,
    window: int = 63,
    lag: int = 1,
    n_states: int = 10,
    step: int = 21,
    n_surrogate: int = 100,
) -> pd.DataFrame:
    """Rolling entropy production with surrogate z-score."""
    dates = states.index
    rows = []
    for end in range(window - 1, len(dates), step):
        start = end - window + 1
        chunk = states.iloc[start : end + 1]
        mu = joint_transition_probs(chunk, lag=lag, n_states=n_states)
        sigma = entropy_production(mu)
        mean_s, std_s, z = reversible_surrogate_pvalue(mu, sigma, n_draws=n_surrogate)
        rows.append(
            {"date": dates[end], "sigma": sigma, "surr_mean": mean_s, "surr_std": std_s, "z": z}
        )
    return pd.DataFrame(rows)


def firm_cumulative_entropy_production(
    states: pd.DataFrame, n_states: int = 10, lag: int = 1
) -> pd.Series:
    """Sigma_i = sum_t F_{q_i(t), q_i(t+lag)} with F_ab = log(mu_ab/mu_ba)."""
    mu = joint_transition_probs(states, lag=lag, n_states=n_states)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((mu > 0) & (mu.T > 0), np.log(mu / mu.T), 0.0)
    a = states.values
    out = {}
    for i, tk in enumerate(states.columns):
        col = a[:, i]
        cum = 0.0
        steps = 0
        for t in range(len(col) - lag):
            s = col[t]
            s2 = col[t + lag]
            if np.isnan(s) or np.isnan(s2):
                continue
            cum += float(f[int(s), int(s2)])
            steps += 1
        out[tk] = cum
    return pd.Series(out, name="firm_sigma").sort_values(ascending=False)


def entropy_lag_profile(
    states: pd.DataFrame, lags: list[int], n_states: int = 10
) -> pd.DataFrame:
    """Return sigma(Delta t) over a set of lags."""
    rows = []
    for lag in lags:
        mu = joint_transition_probs(states, lag=lag, n_states=n_states)
        rows.append({"lag": lag, "sigma": entropy_production(mu)})
    return pd.DataFrame(rows)
