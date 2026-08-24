"""Correlation geometry primitives.

Implements the paper's core rolling-window correlation diagnostics:
- Pearson correlation matrix computation with numeric safety
- Mean off-diagonal correlation
- Market-factor share (largest eigenvalue / N since trace(C) = N)
- Participation ratio PR = (sum lambda_k)^2 / sum(lambda_k^2)
- arccos distance matrix M = arccos(clip(C, -1, 1))
- Perron eigenvalue of M and empirical spectral slope beta over
  K = 2..floor(sqrt(N)) fitted by log-log linear regression on the
  magnitudes of the non-Perron eigenvalues.

# Custom implementations — Context7 confirmed no packaged equivalent for
# these paper-specific spectral diagnostics; numpy/scipy primitives are
# used for the underlying linear algebra.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg
from scipy import stats


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return (A + A.T) / 2 to enforce exact symmetry before eigendecomp."""
    return 0.5 * (matrix + matrix.T)


def pearson_correlation(returns_block: np.ndarray) -> np.ndarray:
    """Compute a Pearson correlation matrix from a (T, N) return block.

    Uses double precision and enforces symmetry with a unit diagonal.
    """
    if returns_block.ndim != 2:
        raise ValueError("returns_block must be 2-D (T, N)")
    x = returns_block.astype(np.float64, copy=False)
    corr = np.corrcoef(x, rowvar=False)
    corr = symmetrize(corr)
    np.fill_diagonal(corr, 1.0)
    return corr


def mean_off_diagonal(corr: np.ndarray) -> float:
    """Mean of off-diagonal correlation entries."""
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(corr[mask].mean())


def eigendecompose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition returning eigenvalues in descending order."""
    m = symmetrize(matrix)
    w, v = linalg.eigh(m)
    # descending
    idx = np.argsort(w)[::-1]
    return w[idx], v[:, idx]


def market_factor_share(eigvals: np.ndarray) -> float:
    """Fraction of variance captured by the leading eigenvector.

    Uses trace(C) = N normalization (i.e. lambda_1 / N).
    """
    n = eigvals.shape[0]
    return float(eigvals[0] / n)


def participation_ratio(eigvals: np.ndarray) -> float:
    """PR = (sum lambda_k)^2 / sum(lambda_k^2). For a correlation matrix
    this simplifies to N^2 / sum(lambda_k^2)."""
    num = float(eigvals.sum()) ** 2
    den = float((eigvals ** 2).sum())
    return num / den


def arccos_distance_matrix(gram: np.ndarray) -> np.ndarray:
    """Elementwise M_ij = arccos(G_ij) with G clamped to [-1, 1]."""
    g = np.clip(gram, -1.0, 1.0)
    m = np.arccos(g)
    m = symmetrize(m)
    np.fill_diagonal(m, 0.0)
    return m


def fit_spectral_slope(
    eigvals_desc: np.ndarray, n_universe: int, drop_perron: bool = True
) -> float:
    """Fit beta from a log-log regression of |lambda_K| vs K.

    Following the paper: K ranges over 2..floor(sqrt(N)) and the leading
    Perron eigenvalue (rank 1) is excluded when ``drop_perron`` is True.
    Eigenvalues are assumed to be sorted in descending algebraic order on
    input, matching :func:`eigendecompose`.
    """
    n = int(n_universe)
    k_max = max(int(np.floor(np.sqrt(n))), 3)
    ranks = np.arange(2, k_max + 1)
    if drop_perron:
        vals = np.abs(eigvals_desc[1 : k_max])
    else:
        vals = np.abs(eigvals_desc[: k_max - 1])
    vals = np.where(vals <= 0, np.finfo(float).tiny, vals)
    if len(vals) < 2:
        return float("nan")
    slope, _, _, _, _ = stats.linregress(np.log(ranks), np.log(vals))
    # paper convention: beta is |slope|
    return float(abs(slope))


@dataclass
class RollingDiagnostic:
    """One row of rolling-window diagnostics."""

    date: pd.Timestamp
    mean_corr: float
    market_share: float
    participation_ratio: float
    lambda1_distance: float
    beta_distance: float


def rolling_diagnostics(
    returns: pd.DataFrame,
    lookback: int,
    residualize_market: bool = False,
) -> pd.DataFrame:
    """Compute daily trailing rolling-correlation-geometry diagnostics.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, index by date, columns by ticker. Universe should
        be fixed (constant columns, complete observations).
    lookback : int
        Trailing window length L in trading days.
    residualize_market : bool
        If True, subtract the market factor before computing distance-
        matrix diagnostics (used by exp_2).

    Returns
    -------
    pd.DataFrame with columns
    ``date, mean_corr, market_share, PR, Lambda1_distance, beta_distance``.
    """
    if lookback < 20:
        raise ValueError("Lookback must be >= 20 trading days")
    r = returns.dropna(how="any").sort_index()
    dates = r.index
    values = r.values.astype(np.float64)
    n_obs, n = values.shape
    if n_obs < lookback:
        raise ValueError(f"Not enough rows ({n_obs}) for lookback {lookback}")

    rows: list[RollingDiagnostic] = []
    for i in range(lookback - 1, n_obs):
        block = values[i - lookback + 1 : i + 1, :]
        c = pearson_correlation(block)
        assert c.shape == (n, n)
        assert np.allclose(np.diag(c), 1.0, atol=1e-8), "diag(C) != 1"
        assert abs(np.trace(c) - n) < 1e-6, "trace(C) != N"

        eigvals_c, eigvecs_c = eigendecompose(c)
        if residualize_market:
            v1 = eigvecs_c[:, 0:1]
            c_used = c - eigvals_c[0] * (v1 @ v1.T)
            # re-normalize to unit-diagonal cosine gram
            diag = np.diag(c_used).copy()
            diag = np.clip(diag, 1e-12, None)
            d_inv = 1.0 / np.sqrt(diag)
            c_used = (c_used * d_inv[:, None]) * d_inv[None, :]
            c_used = symmetrize(c_used)
            np.fill_diagonal(c_used, 1.0)
            eigvals_used, _ = eigendecompose(c_used)
        else:
            c_used = c
            eigvals_used = eigvals_c

        m = arccos_distance_matrix(c_used)
        eigvals_m, _ = eigendecompose(m)

        rows.append(
            RollingDiagnostic(
                date=dates[i],
                mean_corr=mean_off_diagonal(c_used),
                market_share=market_factor_share(eigvals_used),
                participation_ratio=participation_ratio(eigvals_used),
                lambda1_distance=float(eigvals_m[0]),
                beta_distance=fit_spectral_slope(eigvals_m, n_universe=n, drop_perron=True),
            )
        )
    out = pd.DataFrame([r.__dict__ for r in rows])
    out = out.rename(columns={"participation_ratio": "PR"})
    return out


def crisis_calm_vs_crash(
    diagnostics: pd.DataFrame,
    crisis_onset: str | pd.Timestamp,
    calm_years: int = 1,
    crash_months: int = 2,
) -> pd.DataFrame:
    """Compute calm-year vs first-crash-months averages and their delta.

    Returns a wide DataFrame indexed by metric with columns
    ``calm, crash, delta``.
    """
    onset = pd.Timestamp(crisis_onset)
    calm_start = onset - pd.DateOffset(years=calm_years)
    crash_end = onset + pd.DateOffset(months=crash_months)
    df = diagnostics.set_index("date").sort_index()
    calm = df.loc[(df.index >= calm_start) & (df.index < onset)]
    crash = df.loc[(df.index >= onset) & (df.index < crash_end)]
    calm_mean = calm.mean(numeric_only=True)
    crash_mean = crash.mean(numeric_only=True)
    delta = crash_mean - calm_mean
    out = pd.DataFrame({"calm": calm_mean, "crash": crash_mean, "delta": delta})
    return out
