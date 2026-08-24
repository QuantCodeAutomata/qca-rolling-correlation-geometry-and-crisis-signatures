# qca-rolling-correlation-geometry-and-crisis-signatures

Reproduction of seven experiments on rolling correlation geometry of the
S&P 500 across three major crises (dot-com 2001, GFC 2008, Covid 2020):

1. **exp_1** — Rolling correlation geometry & crisis signatures.
2. **exp_2** — Market-factor removal, residual geometry, eigenbasis rotation.
3. **exp_3** — Early-warning ROC/AUC from short-lookback fragility signals.
4. **exp_4** — 3D classical MDS visualization of crisis evolution.
5. **exp_5** — Rank-space Markov chains (return vs. volatility channels).
6. **exp_6** — Entropy production and time-irreversibility in rank dynamics.
7. **exp_7** — Directed influence via transfer entropy over ranks.

## Layout

```
data/            # sp500_tickers.csv, sic_sector_map.csv (small); cache/ is gitignored
src/             # correlation geometry, residual/rotation, rank Markov,
                 # entropy production, transfer entropy, MDS, data loader
exp/             # exp1..exp7 runnable scripts, write to results/
tests/           # pytest tests for src (39 tests)
results/         # CSV/PNG outputs and RESULTS.md
```

## Setup

```bash
export MASSIVE_API_KEY=your_key_here          # primary data source
# optional: cap the universe when iterating
export QCA_SCALE_CAP=60
pip install -r requirements.txt               # or: pip install pandas numpy scipy scikit-learn matplotlib pyinform yfinance massive pytest
```

## Run

```bash
python exp/exp1_correlation_geometry.py
python exp/exp2_residual_rotation.py
python exp/exp3_early_warning.py
python exp/exp4_mds_visualization.py
python exp/exp5_rank_markov.py
python exp/exp6_entropy_production.py
python exp/exp7_transfer_entropy.py
pytest tests/
```

Full metrics and paper-vs-reproduction comparison: [`results/RESULTS.md`](results/RESULTS.md).

## Data

`data_loader.py` uses `massive.RESTClient` for daily adjusted OHLCV; falls
back to `yfinance` for windows preceding Massive coverage (pre-Sept-2003 for
some tickers). Raw downloaded parquet files are cached in `data/cache/` and
are **not** committed. Only the small taxonomy files `sp500_tickers.csv` and
`sic_sector_map.csv` are versioned.
