"""Exp 4: 3D classical MDS visualization of crisis evolution."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    build_returns_panel,
    load_cached_panel,
    cap_universe_by_marketcap,
    load_ticker_universe,
)
from src.mds_visualization import (
    crisis_mds_series,
    forward_return_ranks,
)

from .config import PERIODS, RESULTS_DIR, effective_n


def run_period(period_key: str) -> None:
    period = PERIODS[period_key]
    tickers_meta = load_ticker_universe()
    rets = load_cached_panel(period_key)

    if rets.empty:

        rets = build_returns_panel(

            period.start,

            period.end,

            tickers=tickers_meta["ticker"].tolist(),

            pre_roll_days=180,

        )
    n_eff = effective_n(period, 126)
    keep = cap_universe_by_marketcap(rets, None, tickers_meta, n_keep=n_eff)
    sub = rets[keep].dropna(how="any")

    onset = pd.Timestamp(period.crisis_onset)
    display_start = onset - pd.DateOffset(months=6)
    display_end = onset + pd.DateOffset(months=6)

    # Build snapshot dates ~ every 21 trading days between display bounds.
    idx = sub.index
    mask = (idx >= display_start) & (idx <= display_end)
    display_dates = idx[mask]
    if len(display_dates) == 0:
        return
    snap_dates = [str(display_dates[i].date()) for i in range(0, len(display_dates), 21)]

    frames = crisis_mds_series(
        sub,
        lookback=126,
        crisis_dates=snap_dates,
    )
    # Persist coordinates
    all_rows = []
    for date, coords in frames.items():
        df = pd.DataFrame(coords, columns=["x", "y", "z"])
        df["ticker"] = sub.columns
        df["date"] = str(pd.Timestamp(date).date())
        all_rows.append(df)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            RESULTS_DIR / f"exp4_mds_frames_{period.name}.csv", index=False
        )

    fwd = forward_return_ranks(sub, date=onset, horizon_days=252)
    fwd.to_csv(RESULTS_DIR / f"exp4_fwd_ranks_{period.name}.csv")

    # Static snapshot plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if frames:
            middle = sorted(frames)[len(frames) // 2]
            coords = frames[middle]
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            ranks = fwd.reindex(sub.columns).astype(float)
            colors = 1.0 - (ranks - 1) / max(1, (ranks.max() - 1))
            colors = colors.fillna(0.5).values
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=colors,
                cmap="RdYlGn",
                s=20,
            )
            ax.set_title(f"MDS {period.name} @ {pd.Timestamp(middle).date()}")
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / f"exp4_mds_{period.name}.png", dpi=110)
            plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"[exp4] plot skipped for {period.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", nargs="*", default=list(PERIODS))
    args = parser.parse_args()
    for pk in args.periods:
        print(f"[exp4] running {pk}")
        run_period(pk)


if __name__ == "__main__":
    main()
