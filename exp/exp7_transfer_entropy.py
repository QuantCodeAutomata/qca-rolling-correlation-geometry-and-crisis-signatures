"""Exp 7: Directed influence via transfer entropy across sectors and stocks."""
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
from src.rank_markov import rolling_mean_return, rolling_vol
from src.transfer_entropy import (
    corrected_te,
    net_flow,
    sector_mean_rank_states,
    source_shuffled_bias,
    te_matrix,
    tercile_encode,
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
    sectors = _sector_map(sub.columns)

    for channel, stat in [
        ("ret", rolling_mean_return(sub, 10)),
        ("vol", rolling_vol(sub, 21)),
    ]:
        # Sector-level analysis
        sec_states = sector_mean_rank_states(stat, sectors)
        te_sec = te_matrix(sec_states, k=1)
        bias_sec = source_shuffled_bias(sec_states, k=1, n_shuffles=20)
        te_sec_corr = corrected_te(te_sec, bias_sec)
        nf_sec = net_flow(te_sec_corr)
        te_sec_corr.to_csv(RESULTS_DIR / f"exp7_te_sector_{channel}_{period.name}.csv")
        nf_sec.to_frame("net_flow").to_csv(
            RESULTS_DIR / f"exp7_netflow_sector_{channel}_{period.name}.csv"
        )

        # Stock-level: reduce to top 30 by variance of stat for tractability
        top_cols = stat.dropna().std().sort_values(ascending=False).head(30).index
        stk_states = tercile_encode(stat[top_cols])
        te_stk = te_matrix(stk_states, k=1)
        bias_stk = source_shuffled_bias(stk_states, k=1, n_shuffles=10)
        te_stk_corr = corrected_te(te_stk, bias_stk)
        nf_stk = net_flow(te_stk_corr)
        nf_stk_df = nf_stk.to_frame("net_flow")
        nf_stk_df["sector"] = sectors.reindex(nf_stk_df.index).values
        nf_stk_df.to_csv(
            RESULTS_DIR / f"exp7_netflow_stock_{channel}_{period.name}.csv"
        )

    return {"period": period.name, "n_universe": int(len(sub.columns))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", nargs="*", default=list(PERIODS))
    args = parser.parse_args()
    rows = []
    for pk in args.periods:
        print(f"[exp7] running {pk}")
        rows.append(run_period(pk))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "exp7_manifest.csv", index=False)


if __name__ == "__main__":
    main()
