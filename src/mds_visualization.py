"""3-D classical MDS with Procrustes-aligned frames for crisis animation.

Implements exp_4: classical MDS from the arccos distance matrix and Kabsch/
orthogonal-Procrustes alignment of consecutive frames.

# Custom classical-MDS + Procrustes alignment — Context7 confirmed
# sklearn's MDS is non-metric-oriented; direct numpy implementation is
# preferable to match the paper's exact classical-MDS specification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

from .correlation_geometry import arccos_distance_matrix, pearson_correlation


def classical_mds(distance: np.ndarray, dims: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Classical multidimensional scaling.

    Returns coordinates of shape (N, dims) and the top-``dims`` eigenvalues.
    """
    d = np.asarray(distance, dtype=np.float64)
    n = d.shape[0]
    d2 = d ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    b = 0.5 * (b + b.T)
    w, v = linalg.eigh(b)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    top_w = w[:dims].copy()
    # zero out negative eigenvalues (documented pad)
    top_w = np.where(top_w > 0, top_w, 0.0)
    coords = v[:, :dims] * np.sqrt(top_w)
    return coords, w[:dims]


def kabsch_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate ``source`` onto ``target`` via orthogonal Procrustes.

    Both arrays must have shape (N, d) and be centered externally if
    desired. Reflection is allowed (paper embeddings are up to
    orthogonal transformations including reflection).
    """
    r, _ = linalg.orthogonal_procrustes(source, target)
    return source @ r


def crisis_mds_series(
    returns: pd.DataFrame,
    lookback: int,
    crisis_dates: list[str],
    n_frames: int = 5,
) -> dict[str, np.ndarray]:
    """Compute Procrustes-aligned 3-D MDS coordinates at snapshot dates.

    Returns dictionary keyed by date-string -> (N, 3) coordinate array.
    """
    r = returns.sort_index().dropna(how="any")
    dates = r.index
    values = r.values.astype(np.float64)

    snap_dates = [pd.Timestamp(d) for d in crisis_dates]
    frames: dict[str, np.ndarray] = {}
    prev_coords: np.ndarray | None = None
    for d in snap_dates:
        # find valid rolling date <= d
        pos = dates.get_indexer([d], method="pad")[0]
        if pos < lookback - 1:
            continue
        block = values[pos - lookback + 1 : pos + 1, :]
        c = pearson_correlation(block)
        m = arccos_distance_matrix(c)
        coords, _ = classical_mds(m, dims=3)
        if prev_coords is not None:
            coords = kabsch_align(coords, prev_coords)
        prev_coords = coords
        frames[str(d.date())] = coords
    return frames


def forward_return_ranks(
    returns: pd.DataFrame, date: pd.Timestamp, horizon_days: int = 252
) -> pd.Series:
    """Rank names by realized forward simple return over ``horizon_days``.

    Returns integer ranks in [1, N] with 1 = best performer.
    """
    r = returns.sort_index()
    idx = r.index.get_indexer([pd.Timestamp(date)], method="pad")[0]
    end_idx = min(len(r) - 1, idx + horizon_days)
    fwd = (1.0 + r.iloc[idx + 1 : end_idx + 1]).prod() - 1.0
    ranks = fwd.rank(ascending=False, method="first").astype(int)
    return ranks
