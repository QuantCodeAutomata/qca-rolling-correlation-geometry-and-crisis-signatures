"""Unit tests for exp_2 residual-rotation primitives."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.correlation_geometry import eigendecompose, pearson_correlation
from src.residual_rotation import (
    commutator_norm,
    deflate_market_factor,
    projector,
    projector_drift,
    random_subspace_null,
    residual_gram_and_distance,
    sign_align,
)


def _make_corr(n_dates: int = 500, n_assets: int = 25, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.01, size=n_dates)
    idio = rng.normal(0.0, 0.02, size=(n_dates, n_assets))
    r = market[:, None] + idio
    return pearson_correlation(r)


def test_deflation_reduces_trace_by_lambda1() -> None:
    c = _make_corr()
    n = c.shape[0]
    c_res, v1, lam1 = deflate_market_factor(c)
    assert c_res.shape == c.shape
    assert abs(np.trace(c) - n) < 1e-6
    assert abs(np.trace(c_res) - (n - lam1)) < 1e-6


def test_residual_gram_has_unit_diagonal_and_low_mean_corr() -> None:
    c = _make_corr()
    c_res, _, _ = deflate_market_factor(c)
    g_res, m_res = residual_gram_and_distance(c_res)
    n = g_res.shape[0]
    assert np.allclose(np.diag(g_res), 1.0, atol=1e-10)
    assert np.allclose(np.diag(m_res), 0.0, atol=1e-10)
    # off-diag mean should be near zero after market removal
    mask = ~np.eye(n, dtype=bool)
    assert abs(g_res[mask].mean()) < 0.15


def test_projector_is_idempotent_and_symmetric() -> None:
    c = _make_corr()
    _, v = eigendecompose(c)
    p = projector(v, k=5)
    assert p.shape == (v.shape[0], v.shape[0])
    assert np.allclose(p, p.T, atol=1e-10)
    assert np.allclose(p @ p, p, atol=1e-8)
    # trace of orthogonal projector = k
    assert abs(np.trace(p) - 5) < 1e-6


def test_projector_drift_is_zero_at_reference() -> None:
    c = _make_corr()
    _, v = eigendecompose(c)
    p = projector(v, k=3)
    assert projector_drift(p, p) == 0.0


def test_commutator_zero_when_matrices_equal() -> None:
    c = _make_corr()
    assert commutator_norm(c, c) < 1e-10


def test_sign_align_matches_reference() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=10)
    ref /= np.linalg.norm(ref)
    flipped = -ref
    aligned = sign_align(flipped, ref)
    assert np.allclose(aligned, ref)


def test_random_subspace_null_matches_analytic_formula() -> None:
    n, k = 100, 5
    expected = np.sqrt(2 * k * (1 - k / n))
    assert random_subspace_null(k=k, n=n) == expected
