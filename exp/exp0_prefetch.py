"""Prefetch daily returns panels for the three main periods into parquet.

This is a helper to warm the data-cache before running the seven
experiments. Downloads adjusted daily closes for the top-N tickers
(config-controlled) from Massive over the three period windows plus
pre-roll, computes daily simple returns, and saves them to
``data/cache/panel_<period>.parquet``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import build_returns_panel, load_ticker_universe

from .config import PERIODS

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tickers", type=int, default=80)
    parser.add_argument("--pre-roll", type=int, default=760)
    parser.add_argument("--periods", nargs="*", default=list(PERIODS))
    args = parser.parse_args()

    universe = load_ticker_universe()["ticker"].tolist()[: args.n_tickers]
    for pk in args.periods:
        p = PERIODS[pk]
        out = CACHE / f"panel_{pk}.parquet"
        if out.exists():
            print(f"[exp0] already cached: {out.name}")
            continue
        print(f"[exp0] fetching {pk} {p.start}..{p.end} n={len(universe)}")
        rets = build_returns_panel(
            p.start, p.end, tickers=universe, pre_roll_days=args.pre_roll
        )
        print(f"[exp0]   panel shape = {rets.shape}")
        rets.to_parquet(out)


if __name__ == "__main__":
    main()
