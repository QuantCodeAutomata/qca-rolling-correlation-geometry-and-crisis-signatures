"""Unit tests for Exp 6 entropy production and Exp 7 transfer entropy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.entropy_production import (
    entropy_production,
    joint_transition_probs,
    probability_currents,
)
from src.transfer_entropy import (
    corrected_te,
    net_flow,
    source_shuffled_bias,
    te_matrix,
    tercile_encode,
    transfer_entropy_pair,
)


# ---------- Entropy production ---------------------------------------------


def test_entropy_production_is_zero_for_symmetric_joint() -> None:
    mu = np.array([[0.25, 0.15], [0.15, 0.45]])
    assert entropy_production(mu) == pytest.approx(0.0, abs=1e-12)


def test_entropy_production_positive_for_asymmetric_joint() -> None:
    mu = np.array([[0.20, 0.30], [0.10, 0.40]])
    sig = entropy_production(mu)
    assert sig > 0


def test_probability_currents_antisymmetric() -> None:
    mu = np.array([[0.20, 0.30], [0.10, 0.40]])
    j = probability_currents(mu)
    assert np.allclose(j, -j.T, atol=1e-12)


def test_joint_transition_probs_normalizes_to_one() -> None:
    idx = pd.bdate_range("2000-01-03", periods=200)
    rng = np.random.default_rng(0)
    states = pd.DataFrame(
        rng.integers(0, 3, size=(200, 4)),
        index=idx,
        columns=list("ABCD"),
    )
    mu = joint_transition_probs(states, n_states=3, lag=1)
    assert abs(mu.sum() - 1.0) < 1e-10


# ---------- Transfer entropy ------------------------------------------------


def test_te_zero_when_source_independent_of_target() -> None:
    rng = np.random.default_rng(1)
    src = rng.integers(0, 3, size=2000)
    tgt = rng.integers(0, 3, size=2000)
    te = transfer_entropy_pair(src, tgt, k=1)
    # for independent sequences TE should be near 0 (< noise floor)
    assert te < 0.05


def test_te_positive_when_source_drives_target() -> None:
    rng = np.random.default_rng(2)
    n = 3000
    src = rng.integers(0, 3, size=n)
    tgt = np.zeros(n, dtype=int)
    # target at t+1 copies source at t with high probability
    for t in range(n - 1):
        if rng.random() < 0.85:
            tgt[t + 1] = src[t]
        else:
            tgt[t + 1] = rng.integers(0, 3)
    te_src_to_tgt = transfer_entropy_pair(src, tgt, k=1)
    te_tgt_to_src = transfer_entropy_pair(tgt, src, k=1)
    assert te_src_to_tgt > te_tgt_to_src
    assert te_src_to_tgt > 0.05


def test_tercile_encode_returns_three_classes() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(size=(100, 6)),
        index=pd.bdate_range("2000-01-03", periods=100),
        columns=list("ABCDEF"),
    )
    enc = tercile_encode(df)
    vals = np.unique(enc.values)
    vals = vals[~np.isnan(vals)]
    assert set(vals.astype(int)).issubset({0, 1, 2})


def test_te_matrix_and_net_flow_sum_to_zero() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", periods=400)
    states = pd.DataFrame(
        rng.integers(0, 3, size=(400, 4)),
        index=idx,
        columns=list("ABCD"),
    )
    te = te_matrix(states, k=1)
    assert te.shape == (4, 4)
    nf = net_flow(te)
    assert abs(nf.sum()) < 1e-9


def test_source_shuffled_bias_matches_expected_shape() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", periods=400)
    states = pd.DataFrame(
        rng.integers(0, 3, size=(400, 3)),
        index=idx,
        columns=list("ABC"),
    )
    bias = source_shuffled_bias(states, k=1, n_shuffles=5, seed=0)
    assert bias.shape == (3, 3)


def test_corrected_te_subtracts_bias() -> None:
    te = pd.DataFrame(np.array([[0.0, 0.2], [0.15, 0.0]]), index=list("AB"), columns=list("AB"))
    bias = pd.DataFrame(np.array([[0.0, 0.05], [0.05, 0.0]]), index=list("AB"), columns=list("AB"))
    out = corrected_te(te, bias)
    assert out.loc["A", "B"] == pytest.approx(0.15)
    assert out.loc["B", "A"] == pytest.approx(0.10)
