"""Unit tests for Exp 3 early warning and Exp 4 MDS visualization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.early_warning import (
    equal_weight_market,
    forward_drawdown_label,
    orient_signal,
    roc_curve_for,
)
from src.mds_visualization import classical_mds, kabsch_align


def test_equal_weight_market_is_row_mean() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(100, 5)),
        index=pd.bdate_range("2000-01-03", periods=100),
        columns=list("ABCDE"),
    )
    ew = equal_weight_market(df)
    assert np.allclose(ew.values, df.mean(axis=1).values)


def test_forward_drawdown_label_flags_known_crash() -> None:
    idx = pd.bdate_range("2000-01-03", periods=60)
    r = np.zeros(60)
    r[10:15] = -0.03  # ~15% cumulative drawdown starting at day 10
    ew = pd.Series(r, index=idx)
    labels = forward_drawdown_label(ew, horizon=21, threshold=0.07)
    # observation at day 9 should look forward 21d and see the crash
    assert labels.iloc[9] == 1
    # a label far from the crash should be 0
    assert labels.iloc[0] == 1  # crash still lies within horizon 21 from day 0
    assert labels.iloc[40] == 0


def test_orient_signal_flips_sign_when_needed() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    same = orient_signal(s, higher_is_fragile=True)
    flipped = orient_signal(s, higher_is_fragile=False)
    assert np.allclose(same.values, [1.0, 2.0, 3.0])
    assert np.allclose(flipped.values, [-1.0, -2.0, -3.0])


def test_roc_curve_gives_auc_between_0_and_1() -> None:
    rng = np.random.default_rng(0)
    labels = pd.Series(rng.integers(0, 2, size=200))
    signal = pd.Series(rng.normal(size=200))
    _, _, auc = roc_curve_for(signal, labels)
    assert 0.0 <= auc <= 1.0


def test_perfect_signal_gives_auc_one() -> None:
    labels = pd.Series([0] * 50 + [1] * 50)
    signal = pd.Series(np.concatenate([np.zeros(50), np.ones(50)]))
    _, _, auc = roc_curve_for(signal, labels)
    assert auc == pytest.approx(1.0)


# ---------- Classical MDS ---------------------------------------------------


def test_classical_mds_recovers_pairwise_distances() -> None:
    """For a small Euclidean point cloud MDS should preserve distances."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(15, 3))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    coords, eigs = classical_mds(d, dims=3)
    assert coords.shape == (15, 3)
    # pairwise distances from embedding must approximate the input
    d_hat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    assert np.allclose(d, d_hat, atol=1e-6)
    assert (eigs > -1e-6).all()


def test_kabsch_align_matches_rotated_copy() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20, 3))
    # arbitrary rotation
    theta = 0.7
    r = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    rotated = pts @ r
    aligned = kabsch_align(rotated, pts)
    assert np.allclose(aligned, pts, atol=1e-6)
