"""Unit tests for Exp 5 rank-space Markov chains."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rank_markov import (
    cross_sectional_decile,
    migration_rate,
    mixing_time,
    rolling_mean_return,
    rolling_vol,
    second_eigenvalue_modulus,
    stay_probabilities,
    transition_matrix,
)


def _returns(n_dates: int = 200, n_assets: int = 20, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n_dates)
    return pd.DataFrame(
        rng.normal(0.0, 0.02, size=(n_dates, n_assets)),
        index=idx,
        columns=[f"A{i}" for i in range(n_assets)],
    )


def test_cross_sectional_decile_produces_10_classes() -> None:
    r = _returns()
    stat = rolling_mean_return(r, window=10)
    dec = cross_sectional_decile(stat, n_deciles=10)
    unique = np.unique(dec.dropna(how="all").values)
    unique = unique[~np.isnan(unique)]
    assert set(unique.astype(int)).issubset(set(range(10)))
    # class sizes differ by at most one
    counts = dec.iloc[-1].value_counts()
    assert counts.max() - counts.min() <= 1


def test_transition_matrix_rows_sum_to_one() -> None:
    r = _returns()
    stat = rolling_mean_return(r, window=10)
    dec = cross_sectional_decile(stat, n_deciles=10)
    p = transition_matrix(dec, n_states=10)
    assert p.shape == (10, 10)
    row_sums = p.sum(axis=1)
    # rows that had any transitions must sum to 1; empty rows -> 0 (allowed)
    for rs in row_sums:
        assert (abs(rs - 1.0) < 1e-8) or (abs(rs) < 1e-8)


def test_stay_probabilities_are_diagonal_entries() -> None:
    p = np.array([[0.7, 0.3], [0.1, 0.9]])
    stays = stay_probabilities(p)
    assert np.allclose(stays, [0.7, 0.9])


def test_mixing_time_positive_for_non_trivial_chain() -> None:
    p = np.array([[0.9, 0.1], [0.1, 0.9]])
    lam2 = second_eigenvalue_modulus(p)
    assert 0 < lam2 < 1
    tau = mixing_time(p)
    assert tau > 0


def test_migration_rate_zero_when_states_are_constant() -> None:
    idx = pd.bdate_range("2000-01-03", periods=50)
    states = pd.DataFrame(np.full((50, 5), 3), index=idx, columns=list("ABCDE"))
    mr = migration_rate(states, smooth=5)
    assert np.nanmean(mr) == pytest.approx(0.0, abs=1e-10)


def test_volatility_chain_more_persistent_than_return_chain() -> None:
    """Sanity check that rolling volatility persists longer than 10d mean return
    on synthetic AR(1)-noise. Not tied to a numeric target from the paper."""
    rng = np.random.default_rng(11)
    n_dates, n = 800, 30
    idx = pd.bdate_range("2000-01-03", periods=n_dates)
    # Volatility clusters via multiplicative gamma-modulated noise
    scale = np.abs(rng.normal(1.0, 0.5, size=(n_dates, n))).cumprod(axis=0) ** 0.02
    ret = rng.normal(0.0, 0.02, size=(n_dates, n)) * scale
    df = pd.DataFrame(ret, index=idx, columns=[f"S{i}" for i in range(n)])
    r_stat = rolling_mean_return(df, window=10)
    v_stat = rolling_vol(df, window=21)
    r_dec = cross_sectional_decile(r_stat, n_deciles=10)
    v_dec = cross_sectional_decile(v_stat, n_deciles=10)
    p_r = transition_matrix(r_dec, n_states=10)
    p_v = transition_matrix(v_dec, n_states=10)
    assert second_eigenvalue_modulus(p_v) > second_eigenvalue_modulus(p_r)
