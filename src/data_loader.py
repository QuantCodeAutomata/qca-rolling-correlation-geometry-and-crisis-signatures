"""Data loading utilities.

Fetches daily adjusted OHLCV bars for the S&P 500 universe with Massive API
as the primary source. For historical windows where Massive coverage is
unavailable (Massive daily coverage begins around September 2003), yfinance
is used as an explicit documented fallback (aligned with the project's
data-loading policy).

Outputs are cached as parquet in ``data/cache/`` (which is git-ignored).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MASSIVE_EARLIEST = pd.Timestamp("2003-09-10")


def load_ticker_universe() -> pd.DataFrame:
    """Load the static S&P 500 large-cap ticker list with SIC codes.

    Returns
    -------
    pd.DataFrame with columns ``ticker``, ``name``, ``sic``.
    """
    df = pd.read_csv(DATA_DIR / "sp500_tickers.csv")
    df["sic"] = df["sic"].astype(int)
    return df


def load_sector_map() -> pd.DataFrame:
    """Load SIC-range → custom sector taxonomy used throughout the study."""
    return pd.read_csv(DATA_DIR / "sic_sector_map.csv")


def sic_to_sector(sic: int, sector_map: pd.DataFrame) -> str:
    """Map a SIC code to a sector label using the custom taxonomy."""
    mask = (sector_map["sic_low"] <= sic) & (sic <= sector_map["sic_high"])
    hit = sector_map.loc[mask, "sector"]
    if len(hit) == 0:
        return "Other"
    return str(hit.iloc[0])


def _fetch_massive_single(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily adjusted bars for one ticker via Massive REST."""
    from massive import RESTClient

    key = os.environ.get("MASSIVE_TOKEN") or os.environ.get("MASSIVE_API_KEY")
    client = RESTClient(api_key=key)
    aggs = list(
        client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start,
            to=end,
            limit=50000,
            adjusted=True,
        )
    )
    if not aggs:
        return pd.DataFrame()
    rows = []
    for a in aggs:
        rows.append({"timestamp": a.timestamp, "close": a.close})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(
        "America/New_York"
    ).dt.normalize().dt.tz_localize(None)
    df = df[["date", "close"]].drop_duplicates(subset=["date"]).sort_values("date")
    return df


def _fetch_yahoo_single(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fallback via yfinance for pre-2003 dates. Uses adjusted close.

    Only used to obtain periods that predate Massive's daily coverage. The
    project data policy explicitly permits yfinance as an alternative
    source when Massive is unavailable.
    """
    import yfinance as yf

    yf_ticker = ticker.replace(".", "-")  # yfinance convention: BF.B -> BF-B
    df = yf.download(
        yf_ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df[["Close"]].reset_index().rename(columns={"Date": "date", "Close": "close"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out[["date", "close"]]


def fetch_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
    source: str = "auto",
    sleep_between: float = 0.05,
) -> pd.DataFrame:
    """Fetch daily adjusted close prices for a list of tickers.

    Uses a per-ticker parquet cache under ``data/cache/`` keyed by
    ``ticker__source__start__end.parquet``.

    Parameters
    ----------
    tickers : iterable of str
    start, end : ISO dates ``YYYY-MM-DD`` inclusive on both ends.
    source : ``"massive"``, ``"yahoo"``, or ``"auto"``. ``auto`` uses Massive
        when the window is fully within Massive coverage, otherwise Yahoo.

    Returns
    -------
    Wide DataFrame indexed by date with one column per ticker.
    """
    start_ts = pd.Timestamp(start)

    if source == "auto":
        eff_source = "yahoo" if start_ts < MASSIVE_EARLIEST else "massive"
    else:
        eff_source = source

    frames: dict[str, pd.Series] = {}
    for tk in tickers:
        cache_file = CACHE_DIR / f"{tk.replace('.','_')}__{eff_source}__{start}__{end}.parquet"
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
        else:
            try:
                if eff_source == "massive":
                    df = _fetch_massive_single(tk, start, end)
                else:
                    df = _fetch_yahoo_single(tk, start, end)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {tk} fetch failed ({eff_source}): {exc}")
                df = pd.DataFrame()
            if not df.empty:
                df.to_parquet(cache_file, index=False)
            time.sleep(sleep_between)
        if not df.empty:
            frames[tk] = df.set_index("date")["close"].astype(float)

    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1)
    wide = wide.sort_index()
    return wide


def build_returns_panel(
    period_start: str,
    period_end: str,
    tickers: Iterable[str] | None = None,
    pre_roll_days: int = 520,
    source: str = "auto",
) -> pd.DataFrame:
    """Build a wide dense daily total-return panel for a fixed universe.

    The panel spans from ``period_start - pre_roll_days`` to ``period_end``.
    Only tickers with complete daily price history over the entire span
    (no missing values) are retained. This enforces the paper's constant-
    membership constant-history filter.

    Returns
    -------
    Wide DataFrame of daily simple returns, one column per retained ticker.
    """
    if tickers is None:
        tickers = load_ticker_universe()["ticker"].tolist()

    fetch_start = (pd.Timestamp(period_start) - pd.Timedelta(days=int(pre_roll_days * 1.5))).strftime("%Y-%m-%d")
    prices = fetch_prices(tickers, start=fetch_start, end=period_end, source=source)
    if prices.empty:
        return prices

    prices = prices.loc[fetch_start:period_end]
    # keep only tickers with complete history over the whole span
    keep = [c for c in prices.columns if prices[c].notna().all()]
    prices = prices[keep]
    # daily simple returns
    rets = prices.pct_change().iloc[1:]
    # drop any residual NaN rows
    rets = rets.dropna(how="any")
    return rets


def cap_universe_by_marketcap(
    returns: pd.DataFrame,
    prices: pd.DataFrame | None,
    tickers_meta: pd.DataFrame,
    n_keep: int,
) -> list[str]:
    """Rank stocks by proxy average market-cap and retain the top ``n_keep``.

    A rigorous market-cap proxy requires shares outstanding. Without it we
    use the natural-log average of adjusted close as a mean-size proxy,
    which is monotonic in per-share price and preserves cap-ranking for
    stocks with comparable share counts. This is the paper-agnostic proxy
    consistent with the ``implementation_notes`` guidance to document
    departures.
    """
    if prices is None:
        proxy = returns.abs().mean().rank(ascending=False)
    else:
        proxy = np.log(prices[returns.columns].mean()).rank(ascending=False)
    keep = proxy.sort_values().head(n_keep).index.tolist()
    return keep


def load_cached_panel(period_key: str) -> pd.DataFrame:
    """Load a pre-fetched daily-returns panel written by ``exp0_prefetch``.

    Returns an empty DataFrame if the parquet cache does not yet exist.
    """
    path = CACHE_DIR / f"panel_{period_key}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
