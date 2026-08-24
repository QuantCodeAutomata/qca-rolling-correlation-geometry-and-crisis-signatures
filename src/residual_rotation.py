"""Market-factor removal, residual geometry, and eigenspace rotation.

Implements exp_2 methodology: spectral deflation of the leading eigenmode,
residual observables, sign-aligned leading-eigenvector overlaps, top-K
projector drift, correlation-matrix commutator norm, and residual leading-
eigenvector sector concentration / name-level attribution.

# Custom — Context7 confirmed no packaged equivalent for these paper-
# specific residual-geometry diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

from .correlation_geometry import (
    arccos_distance_matrix,
    eigendecompose,
    fit_spectral_slope,
    mean_off_diagonal,
    participation_ratio,
    pearson_correlation,
    symmetrize,
)


def deflate_market_factor(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return C_res = C - lambda_1 v_1 v_1^T along with (v_1, lambda_1)."""
    w, v = eigendecompose(corr)
    lam1 = float(w[0])
    v1 = v[:, 0]
    c_res = corr - lam1 * np.outer(v1, v1)
    c_res = symmetrize(c_res)
    return c_res, v1, lam1


def residual_gram_and_distance(c_res: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-normalize deflated matrix to a unit-diagonal cosine Gram and
    compute the arccos distance matrix."""
    diag = np.diag(c_res).copy()
    diag = np.clip(diag, 1e-12, None)
    d_inv = 1.0 / np.sqrt(diag)
    g_res = (c_res * d_inv[:, None]) * d_inv[None, :]
    g_res = symmetrize(g_res)
    np.fill_diagonal(g_res, 1.0)
    m_res = arccos_distance_matrix(g_res)
    return g_res, m_res


def sign_align(vec: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip vec's sign if it improves alignment with reference."""
    if float(np.dot(vec, reference)) < 0.0:
        return -vec
    return vec


def projector(eigvecs_desc: np.ndarray, k: int) -> np.ndarray:
    """Rank-K orthogonal projector V_K V_K^T from descending eigvecs."""
    v_k = eigvecs_desc[:, :k]
    return v_k @ v_k.T


def projector_drift(p_ref: np.ndarray, p_t: np.ndarray) -> float:
    """Frobenius norm ||P_t - P_ref||_F."""
    return float(linalg.norm(p_t - p_ref, "fro"))


def commutator_norm(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius norm of the commutator ||[A, B]||_F."""
    ab = a @ b
    ba = b @ a
    return float(linalg.norm(ab - ba, "fro"))


def random_subspace_null(k: int, n: int) -> float:
    """Analytic random-subspace expected projector drift sqrt(2K(1-K/N))."""
    return float(np.sqrt(2.0 * k * (1.0 - k / n)))


def rolling_residual_rotation(
    returns: pd.DataFrame,
    lookback: int,
    anchor_date: str | pd.Timestamp,
    k_projector: int = 5,
) -> pd.DataFrame:
    """Compute residual/rotation diagnostics over a rolling window.

    For each date t after the lookback, compute:
    - mean_corr_res, market_share_res, PR_res, Lambda1_res, beta_res
    - raw_overlap : |<v_1_raw(t), v_1_raw(ref)>|
    - res_overlap : |<v_1_res(t), v_1_res(ref)>|
    - proj_drift_raw_K : ||P_K^raw(t) - P_K^raw(ref)||_F
    - proj_drift_res_K : same for the residual eigenspace
    - commutator : ||[C(ref), C(t)]||_F

    The anchor is the closest rolling date at or before ``anchor_date``.
    """
    r = returns.sort_index().dropna(how="any")
    dates = r.index
    values = r.values.astype(np.float64)
    n_obs, n = values.shape
    anchor_ts = pd.Timestamp(anchor_date)

    # First pass: precompute C(t), residual C_res(t), leading eigenvectors
    stored: dict[pd.Timestamp, dict] = {}
    for i in range(lookback - 1, n_obs):
        block = values[i - lookback + 1 : i + 1, :]
        c = pearson_correlation(block)
        w_c, v_c = eigendecompose(c)
        c_res, v1_raw, lam1 = deflate_market_factor(c)
        g_res, m_res = residual_gram_and_distance(c_res)
        w_g, v_g = eigendecompose(g_res)
        w_m, _ = eigendecompose(m_res)
        stored[dates[i]] = {
            "C": c,
            "w_c": w_c,
            "v_c": v_c,
            "C_res": c_res,
            "G_res": g_res,
            "M_res": m_res,
            "w_g": w_g,
            "v_g": v_g,
            "w_m": w_m,
        }

    # Pick anchor date
    valid_dates = pd.Index([d for d in stored])
    anchor_hits = valid_dates[valid_dates <= anchor_ts]
    if len(anchor_hits) == 0:
        raise ValueError(f"No rolling observation on or before {anchor_ts}")
    t_ref = anchor_hits[-1]
    ref = stored[t_ref]

    # Sign-align eigenvectors sequentially, then anchor overlaps
    prev_raw = ref["v_c"][:, 0].copy()
    prev_res = ref["v_g"][:, 0].copy()
    ref_raw_v1 = prev_raw.copy()
    ref_res_v1 = prev_res.copy()
    p_ref_raw = projector(ref["v_c"], k_projector)
    p_ref_res = projector(ref["v_g"], k_projector)
    c_ref = ref["C"]

    rows = []
    for d in sorted(stored.keys()):
        s = stored[d]
        # sign-align sequentially
        v1_raw = sign_align(s["v_c"][:, 0], prev_raw)
        v1_res = sign_align(s["v_g"][:, 0], prev_res)
        prev_raw = v1_raw
        prev_res = v1_res

        raw_overlap = abs(float(np.dot(v1_raw, ref_raw_v1)))
        res_overlap = abs(float(np.dot(v1_res, ref_res_v1)))

        # projectors from ordered eigvecs; propagate signs first-K
        v_c_signed = s["v_c"].copy()
        v_g_signed = s["v_g"].copy()
        v_c_signed[:, 0] = v1_raw
        v_g_signed[:, 0] = v1_res
        p_raw = projector(v_c_signed, k_projector)
        p_res = projector(v_g_signed, k_projector)

        rows.append(
            {
                "date": d,
                "mean_corr_res": mean_off_diagonal(s["G_res"]),
                "market_share_res": float(s["w_g"][0] / n),
                "PR_res": participation_ratio(s["w_g"]),
                "Lambda1_res": float(s["w_m"][0]),
                "beta_res": fit_spectral_slope(s["w_m"], n_universe=n, drop_perron=True),
                "raw_overlap": raw_overlap,
                "res_overlap": res_overlap,
                "proj_drift_raw": projector_drift(p_ref_raw, p_raw),
                "proj_drift_res": projector_drift(p_ref_res, p_res),
                "commutator": commutator_norm(c_ref, s["C"]),
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["anchor_date"] = str(t_ref)
    out.attrs["random_subspace_null"] = random_subspace_null(k_projector, n)
    out.attrs["random_vector_rms_null"] = float(1.0 / np.sqrt(n))
    return out


def residual_leading_vector(returns: pd.DataFrame, lookback: int, date: pd.Timestamp) -> np.ndarray:
    """Return the sign-anchored residual leading eigenvector at ``date``."""
    r = returns.sort_index().dropna(how="any")
    idx = r.index.get_indexer([pd.Timestamp(date)], method="pad")[0]
    if idx < lookback - 1:
        raise ValueError("Date precedes first rolling window")
    block = r.values[idx - lookback + 1 : idx + 1, :].astype(np.float64)
    c = pearson_correlation(block)
    c_res, _, _ = deflate_market_factor(c)
    g_res, _ = residual_gram_and_distance(c_res)
    w_g, v_g = eigendecompose(g_res)
    return v_g[:, 0]


def sector_concentration(
    v: np.ndarray,
    sectors: pd.Series,
    columns: pd.Index,
) -> pd.DataFrame:
    """Compute per-sector squared-loading mass and concentration ratio.

    Concentration ratio Conc_s = w_s / (n_s / N), where w_s is squared
    loading mass in sector s and n_s is the number of names in s.
    """
    if len(v) != len(columns):
        raise ValueError("Eigenvector length must match columns")
    s = pd.Series(v ** 2, index=columns, name="w")
    df = pd.DataFrame({"w": s, "sector": sectors.reindex(columns).values})
    grouped = df.groupby("sector")["w"].sum().rename("w_s")
    counts = df.groupby("sector").size().rename("n_s")
    n = float(len(v))
    ratio = grouped / (counts / n)
    return pd.DataFrame({"w_s": grouped, "n_s": counts, "conc": ratio, "log2_conc": np.log2(ratio)})


def name_level_movers(
    v_ref: np.ndarray,
    v_peak: np.ndarray,
    columns: pd.Index,
    top: int = 20,
) -> pd.DataFrame:
    """Rank names by absolute loading shift between anchor and peak."""
    delta = v_peak - v_ref
    df = pd.DataFrame({"ticker": columns, "delta": delta, "abs_delta": np.abs(delta)})
    return df.sort_values("abs_delta", ascending=False).head(top).reset_index(drop=True)


def half_shift_date(
    path: pd.DataFrame,
    ticker: str,
    ref_val: float,
    peak_val: float,
) -> pd.Timestamp | None:
    """Return the first date at which ticker's loading crosses 50% of its
    total anchor-to-peak shift.

    ``path`` must have columns ``date`` and one column per ticker giving
    the sign-anchored loading over time.
    """
    if ticker not in path.columns:
        return None
    half = ref_val + 0.5 * (peak_val - ref_val)
    series = path.set_index("date")[ticker]
    if peak_val >= ref_val:
        hits = series[series >= half]
    else:
        hits = series[series <= half]
    if hits.empty:
        return None
    return hits.index[0]
