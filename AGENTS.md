# Repo notes for future agents

## Data pipeline

- Primary: `massive.RESTClient` (env `MASSIVE_API_KEY`). Falls back to
  `yfinance` for pre-Sept-2003 coverage gaps.
- Cache: `data/cache/*.parquet`. **Never** commit — already in `.gitignore`.
- Universe files: `data/sp500_tickers.csv`, `data/sic_sector_map.csv`
  (small, tracked).
- Period cross-sections use ex-post fixed universes with complete return
  histories. `QCA_SCALE_CAP` env var caps universe size for iteration.

## Sub-module contracts

- `src/correlation_geometry.py` — rolling C(t), arccos M(t), participation
  ratio, β via log-log linregress over K = 2..⌊√N⌋. All matrices symmetrised
  and cos entries clamped to [−1, 1].
- `src/residual_rotation.py` — spectral deflation, sign-alignment recursive
  then vs. anchor, projector Frobenius drift D_K and commutator norm.
- `src/rank_markov.py` — 10-day mean-return / 21-day vol ranks → 10 balanced
  deciles. Deterministic tiebreak `method='first'` sorted by ticker.
- `src/entropy_production.py` — pooled joint μ_ab, σ, reversible surrogate
  μ_rev = (μ + μᵀ)/2, per-firm cumulative Σ.
- `src/transfer_entropy.py` — pyinform k=1 pairwise TE with source-shuffled
  surrogate bias subtraction; **`sector_mean_rank_states(stock_ranks, sectors)`
  takes only two args — do not add `n_states=` kwarg**.
- `src/mds_visualization.py` — classical MDS (double-centred Gram, top-3
  positive eigenpairs), Procrustes frame alignment.

## Tests

`pytest tests/` — 39 tests. Cover ranking, correlation geometry, residual
rotation, MDS/early-warning, and entropy/TE modules.

## Crisis dates (all experiments)

- dot-com onset 2001-03-01
- GFC onset 2008-09-15
- Covid onset 2020-03-01
- anchor = onset − 1 calendar year
