"""Early-warning ROC/AUC diagnostics from short-lookback fragility signals.

Implements exp_3: construct short-lookback fragility signals from correlation
geometry, define forward drawdown labels on the fixed-universe equal-weight
market benchmark, and score each signal via ROC/AUC.

# Using sklearn.metrics.roc_auc_score / roc_curve — Context7 confirmed
# these are the standard implementations for the ROC evaluation task.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def equal_weight_market(returns: pd.DataFrame) -> pd.Series:
    """Equal-weighted daily market return series from a fixed universe."""
    return returns.mean(axis=1).rename("r_EW")


def forward_drawdown_label(
    ret_ew: pd.Series,
    horizon: int,
    threshold: float,
) -> pd.Series:
    """Label date t as 1 if the equal-weight market's forward peak-to-trough
    drawdown within the next ``horizon`` trading days exceeds ``threshold``.

    Uses cumulative simple returns.
    """
    ret_ew = ret_ew.sort_index()
    values = ret_ew.values
    n = len(values)
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        j_end = min(n, i + 1 + horizon)
        if j_end - (i + 1) < 1:
            labels[i] = 0
            continue
        segment = values[i + 1 : j_end]
        # cumulative simple returns starting from unit wealth
        wealth = np.cumprod(1.0 + segment)
        running_max = np.maximum.accumulate(wealth)
        dd = 1.0 - wealth / running_max  # positive drawdown fraction
        max_dd = float(dd.max()) if len(dd) else 0.0
        labels[i] = int(max_dd > threshold)
    return pd.Series(labels, index=ret_ew.index, name=f"dd_h{horizon}")


def orient_signal(signal: pd.Series, higher_is_fragile: bool) -> pd.Series:
    """Standardize signal orientation so larger => greater fragility."""
    if higher_is_fragile:
        return signal
    return -signal


def commutator_signal(diag_rot: pd.DataFrame, anchor_col: str = "commutator") -> pd.Series:
    """Extract the commutator signal from a residual/rotation table."""
    s = diag_rot.set_index("date")[anchor_col].rename("commutator")
    return s


def local_rotation_signal(diag_rot: pd.DataFrame, lag: int = 21) -> pd.Series:
    """Local one-month rotation: 1 - aligned overlap over a 21-day lag.

    Uses the residual leading eigenvector overlap propagated by sign
    alignment.
    """
    df = diag_rot.set_index("date").sort_index()
    # 21-trading-day change of proj_drift_res is our documented proxy
    prox = df["proj_drift_res"].diff(lag).rename("rot_1m")
    return prox


@dataclass
class ROCResult:
    signal: str
    horizon: int
    period: str
    auc: float


def score_signals(
    signals: pd.DataFrame,
    labels: pd.Series,
    horizon: int,
    period: str,
) -> pd.DataFrame:
    """Compute AUC for each column of ``signals`` against ``labels``."""
    # Drop duplicate index entries that arise when pooling multiple periods.
    signals = signals[~signals.index.duplicated(keep="first")]
    labels = labels[~labels.index.duplicated(keep="first")]
    common = signals.index.intersection(labels.index)
    y = labels.reindex(common).values
    if len(common) == 0 or y.sum() == 0 or y.sum() == len(y):
        rows = []
        for name in signals.columns:
            rows.append(ROCResult(signal=name, horizon=horizon, period=period, auc=float("nan")))
        return pd.DataFrame([r.__dict__ for r in rows])

    rows = []
    for name in signals.columns:
        s = signals[name].reindex(common)
        m = s.notna() & pd.Series(y, index=common).notna()
        if m.sum() < 30 or pd.Series(y, index=common)[m].nunique() < 2:
            rows.append(ROCResult(signal=name, horizon=horizon, period=period, auc=float("nan")))
            continue
        auc = roc_auc_score(pd.Series(y, index=common)[m].values, s[m].values)
        rows.append(ROCResult(signal=name, horizon=horizon, period=period, auc=float(auc)))
    return pd.DataFrame([r.__dict__ for r in rows])


def roc_curve_for(signal: pd.Series, labels: pd.Series) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, auc) for a single signal."""
    common = signal.index.intersection(labels.index)
    y = labels.reindex(common).values
    s = signal.reindex(common).values
    m = ~np.isnan(s)
    y = y[m]
    s = s[m]
    if len(y) < 30 or np.unique(y).size < 2:
        return np.array([]), np.array([]), float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    return fpr, tpr, float(auc)
