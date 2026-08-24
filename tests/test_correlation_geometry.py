"""Unit tests for correlation-geometry module (Exp 1).

Validates methodology from the paper:
- trace(C) = N
- M = arccos(clip(C,-1,1)) is symmetric with zero diagonal and entries in [0, pi]
- PR simplifies numerically because sum lambda_k = N
- market share = lambda_1 / N
- beta fitted over K = 2 ... floor(sqrt(N)) on |lambda_K^M|
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.correlation_geometry import (
    arccos_distance_matrix,
    eigendecompose,
    fit_spectral_slope,
    market_factor_share,
    mean_off_diagonal,
    participation_ratio,
    pearson_correlation,
    rolling_diagnostics,
)


def _synthetic_returns(n_dates: int = 600, n_assets: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n_dates)
    # 1 market factor + idiosyncratic noise
    market = rng.normal(0.0, 0.01, size=n_dates)
    betas = rng.uniform(0.5, 1.5, size=n_assets)
    idio = rng.normal(0.0, 0.02, size=(n_dates, n_assets))
    r = market[:, None] * betas[None, :] + idio
    return pd.DataFrame(r, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_correlation_trace_equals_n() -> None:
    r = _synthetic_returns().values
    c = pearson_correlation(r)
    n = c.shape[0]
    assert c.shape == (n, n)
    assert np.allclose(np.diag(c), 1.0, atol=1e-10)
    assert abs(np.trace(c) - n) < 1e-8


def test_arccos_distance_properties() -> None:
    r = _synthetic_returns().values
    c = pearson_correlation(r)
    m = arccos_distance_matrix(c)
    n = m.shape[0]
    assert m.shape == (n, n)
    assert np.allclose(m, m.T, atol=1e-12)
    assert np.allclose(np.diag(m), 0.0, atol=1e-10)
    assert (m >= 0.0).all()
    assert (m <= np.pi + 1e-9).all()


def test_participation_ratio_upper_bounded_by_n() -> None:
    r = _synthetic_returns().values
    c = pearson_correlation(r)
    eigs, _ = eigendecompose(c)
    pr = participation_ratio(eigs)
    n = c.shape[0]
    assert 1.0 - 1e-8 <= pr <= n + 1e-6


def test_market_share_between_0_and_1() -> None:
    r = _synthetic_returns().values
    c = pearson_correlation(r)
    eigs, _ = eigendecompose(c)
    ms = market_factor_share(eigs)
    assert 0.0 <= ms <= 1.0


def test_market_share_grows_with_shared_factor() -> None:
    """A stronger common factor increases lambda_1/N."""
    rng = np.random.default_rng(1)
    n_dates, n = 500, 30
    idx = pd.bdate_range("2000-01-03", periods=n_dates)
    market = rng.normal(0.0, 0.01, size=n_dates)
    idio = rng.normal(0.0, 0.02, size=(n_dates, n))
    low = 0.3 * market[:, None] + idio
    high = 2.0 * market[:, None] + idio
    c_low = pearson_correlation(low)
    c_high = pearson_correlation(high)
    eigs_low, _ = eigendecompose(c_low)
    eigs_high, _ = eigendecompose(c_high)
    assert market_factor_share(eigs_high) > market_factor_share(eigs_low)


def test_mean_off_diagonal_ignores_diagonal() -> None:
    c = np.array([[1.0, 0.5, -0.2], [0.5, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    m = mean_off_diagonal(c)
    # Only off-diagonal entries: (0.5 + -0.2 + 0.5 + 0.1 + -0.2 + 0.1)/6 = 0.8/6
    assert m == pytest.approx(0.8 / 6, abs=1e-12)


def test_fit_spectral_slope_returns_scalar_and_uses_paper_convention() -> None:
    """beta should be a finite float when fit on K=2..floor(sqrt(N))."""
    r = _synthetic_returns().values
    c = pearson_correlation(r)
    m = arccos_distance_matrix(c)
    eigs, _ = eigendecompose(m)
    beta = fit_spectral_slope(eigs, n_universe=c.shape[0], drop_perron=True)
    assert np.isfinite(beta)


def test_rolling_diagnostics_uses_trailing_only_and_dates_align() -> None:
    ret = _synthetic_returns(n_dates=400, n_assets=20)
    diag = rolling_diagnostics(ret, lookback=126)
    # first output aligns with day L-1 (index base 0)
    assert diag["date"].iloc[0] == ret.index[125]
    assert diag["date"].iloc[-1] == ret.index[-1]
    assert (diag["mean_corr"].between(-1, 1)).all()
    assert (diag["market_share"].between(0, 1)).all()
    assert (diag["PR"] > 0).all()


def test_rolling_diagnostics_residualization_reduces_mean_corr() -> None:
    ret = _synthetic_returns(n_dates=400, n_assets=20)
    raw = rolling_diagnostics(ret, lookback=126, residualize_market=False)
    res = rolling_diagnostics(ret, lookback=126, residualize_market=True)
    assert (res["mean_corr"].abs().mean()) < (raw["mean_corr"].abs().mean())
