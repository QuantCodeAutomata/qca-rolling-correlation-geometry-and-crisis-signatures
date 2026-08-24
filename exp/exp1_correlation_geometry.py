"""Exp 1: Rolling correlation geometry and crisis signatures.

Generates daily diagnostics tables (mean_corr, market_share, PR,
Lambda1_distance, beta_distance) for each period and lookback plus a
Table-1-style calm-vs-crash summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.correlation_geometry import (
    crisis_calm_vs_crash,
    rolling_diagnostics,
)
from src.data_loader import (
    build_returns_panel,
    cap_universe_by_marketcap,
    fetch_prices,
    load_cached_panel,
    load_ticker_universe,
)

from .config import LOOKBACKS, PERIODS, RESULTS_DIR, SCALE_CAP, effective_n


def run_period(period_key: str, lookbacks: tuple[int, ...]) -> pd.DataFrame:
    period = PERIODS[period_key]
    tickers_meta = load_ticker_universe()
    # Prefer a pre-fetched cached panel; fall back to on-the-fly fetch.
    rets = load_cached_panel(period_key)
    if rets.empty:
        pre_roll_days = int(max(lookbacks) * 1.6) + 60
        rets = build_returns_panel(
            period.start,
            period.end,
            tickers=tickers_meta["ticker"].tolist(),
            pre_roll_days=pre_roll_days,
        )
    if rets.empty:
        raise RuntimeError(f"No data available for period {period_key}")

    prices = None  # cap_universe_by_marketcap gracefully proxies from returns
    summaries = []
    for L in lookbacks:
        n_eff = effective_n(period, L)
        keep = cap_universe_by_marketcap(rets, prices, tickers_meta, n_keep=n_eff)
        sub = rets[keep].dropna(how="any")
        diag = rolling_diagnostics(sub, lookback=L, residualize_market=False)
        diag["period"] = period.name
        diag["lookback"] = L
        out_path = RESULTS_DIR / f"exp1_diagnostics_{period.name}_L{L}.csv"
        diag.to_csv(out_path, index=False)
        summary = crisis_calm_vs_crash(diag, period.crisis_onset).reset_index()
        summary["period"] = period.name
        summary["lookback"] = L
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", nargs="*", default=list(PERIODS))
    parser.add_argument(
        "--lookbacks", nargs="*", type=int, default=list(LOOKBACKS)
    )
    args = parser.parse_args()

    all_summaries = []
    for pk in args.periods:
        print(f"[exp1] running period={pk}")
        s = run_period(pk, tuple(args.lookbacks))
        all_summaries.append(s)
    table1 = pd.concat(all_summaries, ignore_index=True)
    table1.to_csv(RESULTS_DIR / "exp1_table1_calm_vs_crash.csv", index=False)
    print(table1.to_string(index=False))


if __name__ == "__main__":
    main()
