"""Exp 5: Rank-space Markov chains for return and volatility rankings."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data_loader import (
    build_returns_panel,
    load_cached_panel,
    cap_universe_by_marketcap,
    load_sector_map,
    load_ticker_universe,
    sic_to_sector,
)
from src.rank_markov import (
    cross_sectional_decile,
    independent_return_null,
    migration_rate,
    mixing_time,
    rolling_mean_return,
    rolling_transition_diagnostics,
    rolling_vol,
    second_eigenvalue_modulus,
    sector_coherence,
    stay_probabilities,
    transition_matrix,
)

from .config import PERIODS, RESULTS_DIR, effective_n


def _sector_map(cols: pd.Index) -> pd.Series:
    meta = load_ticker_universe().set_index("ticker")
    sm = load_sector_map()
    return pd.Series(
        {tk: sic_to_sector(int(meta.loc[tk, "sic"]), sm) for tk in cols},
        name="sector",
    )


def run_period(period_key: str) -> dict:
    period = PERIODS[period_key]
    tickers_meta = load_ticker_universe()
    rets = load_cached_panel(period_key)

    if rets.empty:

        rets = build_returns_panel(

            period.start,

            period.end,

            tickers=tickers_meta["ticker"].tolist(),

            pre_roll_days=60,

        )
    n_eff = effective_n(period, 504)
    keep = cap_universe_by_marketcap(rets, None, tickers_meta, n_keep=n_eff)
    sub = rets[keep].dropna(how="any")

    ret_stat = rolling_mean_return(sub, window=10)
    vol_stat = rolling_vol(sub, window=21)

    ret_dec = cross_sectional_decile(ret_stat)
    vol_dec = cross_sectional_decile(vol_stat)

    P_ret = transition_matrix(ret_dec, n_states=10)
    P_vol = transition_matrix(vol_dec, n_states=10)

    pd.DataFrame(P_ret).to_csv(RESULTS_DIR / f"exp5_P_ret_{period.name}.csv")
    pd.DataFrame(P_vol).to_csv(RESULTS_DIR / f"exp5_P_vol_{period.name}.csv")

    stats = {
        "period": period.name,
        "ret_lambda2": second_eigenvalue_modulus(P_ret),
        "vol_lambda2": second_eigenvalue_modulus(P_vol),
        "ret_mixing_time": mixing_time(P_ret),
        "vol_mixing_time": mixing_time(P_vol),
        "ret_mean_stay": float(np.mean(stay_probabilities(P_ret))),
        "vol_mean_stay": float(np.mean(stay_probabilities(P_vol))),
    }

    ret_roll = rolling_transition_diagnostics(ret_dec, window=252, n_states=10)
    vol_roll = rolling_transition_diagnostics(vol_dec, window=252, n_states=10)
    ret_roll.to_csv(RESULTS_DIR / f"exp5_ret_rolling_{period.name}.csv", index=False)
    vol_roll.to_csv(RESULTS_DIR / f"exp5_vol_rolling_{period.name}.csv", index=False)

    ret_mig = migration_rate(ret_dec)
    vol_mig = migration_rate(vol_dec)
    ret_mig.to_csv(RESULTS_DIR / f"exp5_ret_migration_{period.name}.csv")
    vol_mig.to_csv(RESULTS_DIR / f"exp5_vol_migration_{period.name}.csv")

    sectors = _sector_map(sub.columns)
    ret_same, ret_all = sector_coherence(ret_dec, sectors)
    vol_same, vol_all = sector_coherence(vol_dec, sectors)
    stats["ret_same_sector_corr"] = ret_same
    stats["ret_all_pair_corr"] = ret_all
    stats["vol_same_sector_corr"] = vol_same
    stats["vol_all_pair_corr"] = vol_all

    # Independent null: shuffle returns in time, rebuild rolling stat, transition matrix
    P_ret_null = independent_return_null(
        sub, lambda r: rolling_mean_return(r, window=10), n_states=10
    )
    P_vol_null = independent_return_null(
        sub, lambda r: rolling_vol(r, window=21), n_states=10
    )
    stats["ret_null_lambda2"] = second_eigenvalue_modulus(P_ret_null)
    stats["vol_null_lambda2"] = second_eigenvalue_modulus(P_vol_null)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", nargs="*", default=list(PERIODS))
    args = parser.parse_args()
    rows = []
    for pk in args.periods:
        print(f"[exp5] running {pk}")
        rows.append(run_period(pk))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "exp5_summary.csv", index=False)


if __name__ == "__main__":
    main()
