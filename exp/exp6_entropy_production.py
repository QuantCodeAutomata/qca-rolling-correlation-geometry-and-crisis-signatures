"""Exp 6: Entropy production and time-irreversibility in rank dynamics."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data_loader import (
    build_returns_panel,
    load_cached_panel,
    cap_universe_by_marketcap,
    load_ticker_universe,
)
from src.entropy_production import (
    entropy_lag_profile,
    entropy_production,
    firm_cumulative_entropy_production,
    joint_transition_probs,
    probability_currents,
    reversible_surrogate_pvalue,
    rolling_entropy_production,
)
from src.rank_markov import (
    cross_sectional_decile,
    rolling_mean_return,
    rolling_vol,
)

from .config import PERIODS, RESULTS_DIR, effective_n


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

    ret_dec = cross_sectional_decile(rolling_mean_return(sub, 10))
    vol_dec = cross_sectional_decile(rolling_vol(sub, 21))

    mu_ret = joint_transition_probs(ret_dec, lag=1, n_states=10)
    mu_vol = joint_transition_probs(vol_dec, lag=1, n_states=10)

    sigma_ret = entropy_production(mu_ret)
    sigma_vol = entropy_production(mu_vol)

    _, _, z_ret = reversible_surrogate_pvalue(mu_ret, sigma_ret, n_draws=200)
    _, _, z_vol = reversible_surrogate_pvalue(mu_vol, sigma_vol, n_draws=200)

    J_ret = probability_currents(mu_ret)
    J_vol = probability_currents(mu_vol)
    pd.DataFrame(J_ret).to_csv(RESULTS_DIR / f"exp6_J_ret_{period.name}.csv")
    pd.DataFrame(J_vol).to_csv(RESULTS_DIR / f"exp6_J_vol_{period.name}.csv")

    lag_ret = entropy_lag_profile(ret_dec, lags=range(1, 21), n_states=10)
    lag_vol = entropy_lag_profile(vol_dec, lags=range(1, 21), n_states=10)
    lag_ret.to_csv(RESULTS_DIR / f"exp6_lag_ret_{period.name}.csv", index=False)
    lag_vol.to_csv(RESULTS_DIR / f"exp6_lag_vol_{period.name}.csv", index=False)

    roll_vol = rolling_entropy_production(vol_dec, window=63, n_states=10)
    roll_vol.to_csv(RESULTS_DIR / f"exp6_rolling_vol_{period.name}.csv", index=False)

    firm_sigma = firm_cumulative_entropy_production(vol_dec, n_states=10)
    firm_sigma.to_csv(RESULTS_DIR / f"exp6_firm_sigma_vol_{period.name}.csv")

    return {
        "period": period.name,
        "sigma_ret": float(sigma_ret),
        "sigma_vol": float(sigma_vol),
        "z_ret": float(z_ret),
        "z_vol": float(z_vol),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", nargs="*", default=list(PERIODS))
    args = parser.parse_args()
    rows = []
    for pk in args.periods:
        print(f"[exp6] running {pk}")
        rows.append(run_period(pk))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "exp6_summary.csv", index=False)


if __name__ == "__main__":
    main()
