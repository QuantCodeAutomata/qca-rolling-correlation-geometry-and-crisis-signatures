"""Exp 2: Market-factor removal, residual geometry, and eigenbasis rotation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.correlation_geometry import (
    eigendecompose,
    pearson_correlation,
    rolling_diagnostics,
)
from src.data_loader import (
    build_returns_panel,
    load_cached_panel,
    cap_universe_by_marketcap,
    load_sector_map,
    load_ticker_universe,
    sic_to_sector,
)
from src.residual_rotation import (
    deflate_market_factor,
    name_level_movers,
    residual_gram_and_distance,
    residual_leading_vector,
    rolling_residual_rotation,
    sector_concentration,
    sign_align,
)

from .config import PERIODS, RESULTS_DIR, effective_n


def _sector_series(cols: pd.Index) -> pd.Series:
    tickers_meta = load_ticker_universe().set_index("ticker")
    sm = load_sector_map()
    return pd.Series(
        {tk: sic_to_sector(int(tickers_meta.loc[tk, "sic"]), sm) for tk in cols},
        name="sector",
    )


def run_period(period_key: str, lookback: int = 504) -> dict:
    period = PERIODS[period_key]
    tickers_meta = load_ticker_universe()
    pre_roll = int(lookback * 1.6) + 260  # 1yr for anchor
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
    n_eff = effective_n(period, lookback)
    keep = cap_universe_by_marketcap(rets, None, tickers_meta, n_keep=n_eff)
    sub = rets[keep].dropna(how="any")

    # raw + residual diagnostics
    raw_diag = rolling_diagnostics(sub, lookback=lookback, residualize_market=False)
    res_diag = rolling_diagnostics(sub, lookback=lookback, residualize_market=True)
    raw_diag.to_csv(RESULTS_DIR / f"exp2_raw_diag_{period.name}.csv", index=False)
    res_diag.to_csv(RESULTS_DIR / f"exp2_res_diag_{period.name}.csv", index=False)

    onset = pd.Timestamp(period.crisis_onset)
    anchor = onset - pd.DateOffset(years=1)
    rot = rolling_residual_rotation(
        sub, lookback=lookback, anchor_date=anchor, k_projector=5
    )
    rot.to_csv(RESULTS_DIR / f"exp2_rotation_{period.name}.csv", index=False)

    # Peak rotation date: minimum residual overlap in the crash window
    crash_end = onset + pd.DateOffset(months=6)
    rot_dates = pd.to_datetime(rot["date"])
    mask = (rot_dates >= onset) & (rot_dates <= crash_end)
    if mask.any():
        peak_idx = rot.loc[mask, "res_overlap"].idxmin()
        peak_date = pd.Timestamp(rot.loc[peak_idx, "date"])
    else:
        peak_date = onset

    # Anchor + peak residual leading vectors
    v_ref = residual_leading_vector(sub, lookback, anchor)
    v_peak = residual_leading_vector(sub, lookback, peak_date)
    v_peak = sign_align(v_peak, v_ref)

    sectors = _sector_series(sub.columns)
    conc = sector_concentration(v_ref, sectors, sub.columns)
    conc.to_csv(RESULTS_DIR / f"exp2_sector_conc_ref_{period.name}.csv", index=False)
    conc_peak = sector_concentration(v_peak, sectors, sub.columns)
    conc_peak.to_csv(
        RESULTS_DIR / f"exp2_sector_conc_peak_{period.name}.csv", index=False
    )

    movers = name_level_movers(v_ref, v_peak, sub.columns, top=20)
    movers["peak_date"] = str(peak_date.date())
    movers["anchor_date"] = str(anchor.date())
    movers.to_csv(RESULTS_DIR / f"exp2_movers_{period.name}.csv", index=False)

    return {
        "period": period_key,
        "anchor": str(anchor.date()),
        "peak_date": str(peak_date.date()),
        "n_universe": int(len(sub.columns)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", nargs="*", default=list(PERIODS))
    parser.add_argument("--lookback", type=int, default=504)
    args = parser.parse_args()

    manifest = []
    for pk in args.periods:
        print(f"[exp2] running period={pk}")
        manifest.append(run_period(pk, lookback=args.lookback))
    pd.DataFrame(manifest).to_csv(RESULTS_DIR / "exp2_manifest.csv", index=False)


if __name__ == "__main__":
    main()
