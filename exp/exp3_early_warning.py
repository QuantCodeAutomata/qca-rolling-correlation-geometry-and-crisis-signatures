"""Exp 3: Early-warning ROC/AUC diagnostics at L=126."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.correlation_geometry import rolling_diagnostics
from src.data_loader import (
    build_returns_panel,
    load_cached_panel,
    cap_universe_by_marketcap,
    load_ticker_universe,
)
from src.early_warning import (
    equal_weight_market,
    forward_drawdown_label,
    orient_signal,
    score_signals,
)
from src.residual_rotation import rolling_residual_rotation

from .config import PERIODS, RESULTS_DIR, effective_n


def build_signals(returns: pd.DataFrame, period_crisis: str) -> pd.DataFrame:
    diag = rolling_diagnostics(returns, lookback=126, residualize_market=False)
    diag = diag.set_index("date")
    onset = pd.Timestamp(period_crisis)
    anchor = onset - pd.DateOffset(years=1)
    rot = rolling_residual_rotation(returns, lookback=126, anchor_date=anchor, k_projector=5)
    rot = rot.set_index("date")

    signals = pd.DataFrame(index=diag.index)
    signals["mean_corr"] = orient_signal(diag["mean_corr"], higher_is_fragile=True)
    signals["market_share"] = orient_signal(diag["market_share"], higher_is_fragile=True)
    signals["neg_PR"] = orient_signal(-diag["PR"], higher_is_fragile=True)
    # Local rotation: 21-day change in residual projector drift
    loc_rot = rot["proj_drift_res"].diff(21)
    signals["local_rot"] = orient_signal(loc_rot.reindex(signals.index), True)
    signals["commutator"] = orient_signal(
        rot["commutator"].reindex(signals.index), True
    )
    return signals.dropna(how="all")


def run_period(period_key: str) -> pd.DataFrame:
    period = PERIODS[period_key]
    tickers_meta = load_ticker_universe()
    pre_roll = 260
    rets = load_cached_panel(period_key)

    if rets.empty:

        rets = load_cached_panel(period_key)


        if rets.empty:


            rets = build_returns_panel(


                period.start,


                period.end,


                tickers=tickers_meta["ticker"].tolist(),


                pre_roll_days=pre_roll,


            )
    n_eff = effective_n(period, 126)
    keep = cap_universe_by_marketcap(rets, None, tickers_meta, n_keep=n_eff)
    sub = rets[keep].dropna(how="any")

    mkt_ew = equal_weight_market(sub)
    signals = build_signals(sub, period.crisis_onset)
    signals.to_csv(RESULTS_DIR / f"exp3_signals_{period.name}.csv")

    rows = []
    for h, thresh in [(21, 0.07), (63, 0.10)]:
        labels = forward_drawdown_label(mkt_ew, horizon=h, threshold=thresh)
        res = score_signals(signals, labels, horizon=h, period=period.name)
        rows.append(res)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", nargs="*", default=list(PERIODS))
    args = parser.parse_args()

    per_period = []
    all_signals = []
    all_labels_21 = []
    all_labels_63 = []
    for pk in args.periods:
        print(f"[exp3] running period={pk}")
        per_period.append(run_period(pk))
        # Rebuild for pooled: read cached signals
        sig = pd.read_csv(
            RESULTS_DIR / f"exp3_signals_{pk}.csv", index_col=0, parse_dates=[0]
        )
        # labels have to be rebuilt to remain aligned; simplest: rerun
        period = PERIODS[pk]
        tickers_meta = load_ticker_universe()
        rets = build_returns_panel(
            period.start,
            period.end,
            tickers=tickers_meta["ticker"].tolist(),
            pre_roll_days=260,
        )
        n_eff = effective_n(period, 126)
        keep = cap_universe_by_marketcap(rets, None, tickers_meta, n_keep=n_eff)
        sub = rets[keep].dropna(how="any")
        mkt = equal_weight_market(sub)
        all_signals.append(sig)
        all_labels_21.append(forward_drawdown_label(mkt, 21, 0.07))
        all_labels_63.append(forward_drawdown_label(mkt, 63, 0.10))

    perf = pd.concat(per_period, ignore_index=True)
    perf.to_csv(RESULTS_DIR / "exp3_period_auc.csv", index=False)

    # pooled AUC
    pooled_sig = pd.concat(all_signals)
    for h, labs in [(21, all_labels_21), (63, all_labels_63)]:
        pooled_lab = pd.concat(labs)
        pooled = score_signals(pooled_sig, pooled_lab, horizon=h, period="pooled")
        pooled.to_csv(RESULTS_DIR / f"exp3_pooled_auc_h{h}.csv", index=False)
        print(f"[exp3] pooled h={h}")
        print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
