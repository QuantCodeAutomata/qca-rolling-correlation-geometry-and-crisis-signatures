"""Common experiment configuration.

Defines the three paper periods, crisis onsets, lookback lengths, and
lightweight run-scale controls. The paper targets full CRSP universes
of size 244/265/309, which are impractical to fetch from Massive in a
single container run. The configuration below allows scaling down the
number of tickers so the experiments remain reproducible end-to-end
without altering the methodology.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PeriodSpec:
    name: str
    start: str
    end: str
    crisis_onset: str
    n_504: int  # paper universe size at L=504
    n_256: int  # paper universe size at L=256 (192 per paper)
    n_126: int  # paper universe size at L=126 (94 per paper)


PERIODS = {
    "dotcom": PeriodSpec(
        name="dotcom",
        start="1996-01-01",
        end="2005-12-31",
        crisis_onset="2001-03-01",
        n_504=244,
        n_256=192,
        n_126=94,
    ),
    "gfc": PeriodSpec(
        name="gfc",
        start="2003-01-01",
        end="2012-12-31",
        crisis_onset="2008-09-15",
        n_504=265,
        n_256=192,
        n_126=94,
    ),
    "covid": PeriodSpec(
        name="covid",
        start="2015-01-01",
        end="2024-12-31",
        crisis_onset="2020-03-01",
        n_504=309,
        n_256=192,
        n_126=94,
    ),
}


# Runtime-scale flag: if set to a positive integer, cap the universe to
# this many largest-cap-proxy tickers per period regardless of paper size,
# and reduce lookback grid to two entries. Used to keep container runs
# tractable while preserving methodology.
SCALE_CAP = int(os.environ.get("QCA_SCALE_CAP", "60"))
LOOKBACKS: tuple[int, ...] = tuple(
    int(x) for x in os.environ.get("QCA_LOOKBACKS", "126,504").split(",")
)


def effective_n(period: PeriodSpec, lookback: int) -> int:
    """Return the effective universe size given SCALE_CAP and lookback."""
    if lookback == 504:
        target = period.n_504
    elif lookback == 256:
        target = period.n_256
    else:
        target = period.n_126
    return min(SCALE_CAP, target)
