"""Directed influence via transfer entropy.

Implements exp_7: transfer-entropy estimation on tercile-coded rank state
sequences at sector and stock level, source-shuffled surrogate bias
correction, and net-flow rankings.

# Using pyinform.transferentropy — Context7 recommended pyinform for
# discrete transfer-entropy estimation on integer-coded sequences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _te_manual(source: np.ndarray, target: np.ndarray, k: int, base: str = "nat") -> float:
    """Compute one-step transfer entropy TE(S->T) with history length k.

    Uses empirical joint / conditional counts. Fallback in case pyinform
    is unavailable at runtime.
    """
    src = np.asarray(source, dtype=int)
    tgt = np.asarray(target, dtype=int)
    if src.shape != tgt.shape:
        raise ValueError("source and target must share length")
    n = len(src)
    if n <= k + 1:
        return 0.0
    from collections import Counter

    joint_tts_s = Counter()  # (t_hist, s_hist, t_next)
    joint_tts = Counter()  # (t_hist, s_hist)
    joint_tt = Counter()  # (t_hist, t_next)
    joint_t = Counter()  # (t_hist,)
    for t in range(k, n - 1):
        t_hist = tuple(tgt[t - k + 1 : t + 1])
        s_hist = tuple(src[t - k + 1 : t + 1])
        t_next = tgt[t + 1]
        joint_tts_s[(t_hist, s_hist, t_next)] += 1
        joint_tts[(t_hist, s_hist)] += 1
        joint_tt[(t_hist, t_next)] += 1
        joint_t[(t_hist,)] += 1
    total = sum(joint_tts_s.values())
    if total == 0:
        return 0.0
    te = 0.0
    for (t_hist, s_hist, t_next), cnt in joint_tts_s.items():
        p_full = cnt / total
        p_ts = joint_tts[(t_hist, s_hist)] / total
        p_tt = joint_tt[(t_hist, t_next)] / total
        p_t = joint_t[(t_hist,)] / total
        num = p_full * p_t
        den = p_ts * p_tt
        if num > 0 and den > 0:
            te += p_full * np.log(num / den)
    if base == "bits":
        te /= np.log(2)
    return float(te)


def transfer_entropy_pair(source: np.ndarray, target: np.ndarray, k: int = 1) -> float:
    """Wrapper that prefers pyinform when available."""
    try:
        from pyinform import transfer_entropy as pi_te

        return float(pi_te(list(source.astype(int)), list(target.astype(int)), k=k))
    except Exception:
        return _te_manual(source, target, k=k)


def tercile_encode(rank_stat: pd.DataFrame) -> pd.DataFrame:
    """Encode each daily cross-sectional rank into a balanced tercile 0/1/2."""
    def _row(row: pd.Series) -> pd.Series:
        s = row.dropna().sort_values(kind="mergesort")
        n = len(s)
        if n < 3:
            return pd.Series(index=row.index, dtype="float64")
        boundaries = np.linspace(0, n, 4).astype(int)
        vals = np.zeros(n, dtype=int)
        for d in range(3):
            vals[boundaries[d] : boundaries[d + 1]] = d
        out = pd.Series(vals, index=s.index, dtype=float)
        return out.reindex(row.index)

    return rank_stat.apply(_row, axis=1)


def sector_mean_rank_states(
    stock_ranks: pd.DataFrame, sectors: pd.Series
) -> pd.DataFrame:
    """Sector-level tercile states from mean of stock ranks by sector."""
    tick_to_sec = sectors.reindex(stock_ranks.columns)
    sec_series = {}
    for sec, group in tick_to_sec.groupby(tick_to_sec):
        if pd.isna(sec) or len(group) == 0:
            continue
        cols = group.index.tolist()
        sec_series[str(sec)] = stock_ranks[cols].mean(axis=1)
    sec_df = pd.DataFrame(sec_series)
    # tercile encode per date across sectors
    return tercile_encode(sec_df)


def te_matrix(states: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """Pairwise directed TE matrix: entry [i, j] is TE(i -> j)."""
    names = list(states.columns)
    n = len(names)
    m = np.zeros((n, n))
    arr = states.values
    for i in range(n):
        s_i = arr[:, i]
        mask_i = ~np.isnan(s_i)
        for j in range(n):
            if i == j:
                continue
            s_j = arr[:, j]
            mask = mask_i & ~np.isnan(s_j)
            if mask.sum() < 30:
                continue
            m[i, j] = transfer_entropy_pair(s_i[mask], s_j[mask], k=k)
    return pd.DataFrame(m, index=names, columns=names)


def source_shuffled_bias(
    states: pd.DataFrame, k: int = 1, n_shuffles: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Estimate finite-sample TE bias by source-shuffling each pair."""
    rng = np.random.default_rng(seed)
    names = list(states.columns)
    n = len(names)
    accum = np.zeros((n, n))
    arr = states.values
    for _ in range(n_shuffles):
        for i in range(n):
            s_i = arr[:, i]
            mask_i = ~np.isnan(s_i)
            s_i_shuf = s_i.copy()
            s_i_shuf[mask_i] = rng.permutation(s_i[mask_i])
            for j in range(n):
                if i == j:
                    continue
                s_j = arr[:, j]
                mask = mask_i & ~np.isnan(s_j)
                if mask.sum() < 30:
                    continue
                accum[i, j] += transfer_entropy_pair(s_i_shuf[mask], s_j[mask], k=k)
    return pd.DataFrame(accum / max(n_shuffles, 1), index=names, columns=names)


def corrected_te(te: pd.DataFrame, bias: pd.DataFrame) -> pd.DataFrame:
    """Bias-corrected TE matrix."""
    return te.subtract(bias, fill_value=0.0)


def net_flow(te: pd.DataFrame) -> pd.Series:
    """Sector/stock net flow = sum_j TE(s -> j) - sum_j TE(j -> s)."""
    out = te.sum(axis=1) - te.sum(axis=0)
    return out.rename("net_flow").sort_values(ascending=False)
