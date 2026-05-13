# Path 2 — Build a CTA-style 25-instrument basket

**Status:** Deferred. We chose Path 1 (productionize gold + QQQ TSM with futures and
leverage) first. This document captures Path 2 so it isn't lost.

**Goal:** Replicate the structure that small commodity-trading-advisor funds run:
~25 diversified instruments across five macro factors, multi-horizon momentum
signals, and risk-parity weighting. Realistic outcome: portfolio Sharpe ~0.9-1.1
with max drawdown ~15% — substantially better than the 6-asset Sharpe 0.47 we
have today.

**Effort estimate:** ~15 hours of careful work plus 30 min of data downloads.

## Why this works

Theoretical Sharpe gain from N independent bets scales with √N. Going from
6 → 25 assets gives √(25/6) ≈ 2.0× under perfect independence. Realized cross-
asset correlation cuts that to ~1.5×, but starting from our current 0.47, that
lands around 0.7-1.0 — comparable to what AQR, Man AHL, and Winton report
publicly for their managed-futures funds.

Hurst/Ooi/Pedersen (2017) "A Century of Evidence on Trend-Following Investing"
documents Sharpe ~1.0 for diversified TSM portfolios going back to 1880, with
consistent performance across crises (1929, 2000, 2008). The strategy class is
about as well-validated as anything in systematic finance.

## Universe (25 instruments across 5 factors)

Each asset class needs ≥4 instruments so no single name dominates. All daily
bars, vol-targeted to ~6% per asset (so 25 × 6% / √25 ≈ 6% portfolio vol),
combined under risk-parity weighting.

### Equity indices (6)
- **SPY** — US large cap (have)
- **QQQ** — US tech (have)
- **IWM** — US small cap (Russell 2000)
- **EFA** — Developed ex-US (have)
- **EEM** — Emerging markets (have)
- **EWJ** — Japan (decorrelates from US in Asian sessions)

### Fixed income (5)
- **TLT** — 20+ year US treasuries (have)
- **IEF** — 7-10 year US treasuries (have)
- **SHY** — 1-3 year US treasuries (very short duration — different factor)
- **HYG** — US high yield credit (have)
- **EMB** — Emerging market bonds

### Currencies (5)
- **USDJPY** (`JPY=X` on yfinance)
- **GBPUSD** (`GBPUSD=X`)
- **AUDUSD** (`AUDUSD=X`) — commodity-linked
- **USDCAD** (`CAD=X`) — oil-linked
- **USDCHF** (`CHF=X`) — safe-haven

### Commodities (6)
- **GLD** — Gold (have)
- **SLV** — Silver
- **USO** — Crude oil
- **DBA** — Agriculture basket
- **CPER** — Copper
- **PALL** or **PPLT** — Platinum/palladium (diversifies precious metals)

### Volatility / alternatives (3)
- **VNQ** — US REITs
- **GSG** — Broad commodities
- **VXX** — VIX futures (but careful: contango decay; consider skipping)

## Required code changes

### 1. Multi-horizon momentum (`backtest/tsm.py`)

Replace the single 12-1m signal with a blend across multiple horizons:

```python
def multi_horizon_signal(close, horizons=[63, 126, 252], skip_days=21):
    """Average rank across multiple lookback windows. Equivalent to the
    Asness/Moskowitz/Pedersen multi-horizon momentum factor."""
    signals = [tsm_signal(close, h, skip_days) for h in horizons]
    return sum(signals) / len(signals)  # in [-1, 1]
```

Expected lift: +0.1-0.3 Sharpe per asset.

### 2. Risk-parity weighting (`backtest/tsm.py:simulate_tsm_portfolio`)

Replace equal-weight with weights ∝ 1 / realized_vol_of_TSM_returns:

```python
weighting="inverse_vol"  # new parameter
# At each rebalance, weight_i = (1/vol_i) / sum_j(1/vol_j)
# where vol_i is trailing 60-day std of asset_i's TSM return stream
```

This automatically de-weights noisy assets (HYG, EFA) and over-weights stable
ones (gold, QQQ).

### 3. Live signal threshold

Only take positions when |signal magnitude / asset vol| exceeds a Z-threshold
(e.g., 0.3). Filters out weak signals during regime transitions. Adds ~0.05
Sharpe but materially reduces whipsaw.

### 4. Drawdown-based de-risking

Track rolling 6-month portfolio drawdown. If > 10%, halve all positions until
DD recovers. Doesn't improve Sharpe but cuts max DD ~30% — makes the strategy
psychologically trade-able.

## Data acquisition

Most yfinance can fetch. FX needs `=X` suffix. EMB and SHY are ETFs so trivial.
Add to `scripts/download_universe.sh` and re-run.

## Backtest sequence

1. Download all 25 daily CSVs (extend `scripts/download_universe.sh`)
2. Run per-asset TSM sensitivity sweeps — confirm each asset's edge profile
3. Run portfolio with risk-parity weighting + multi-horizon signals
4. Sensitivity test the *portfolio* across signal blends and rebalance frequencies
5. Compare to a buy-and-hold equal-weight benchmark of the same universe

## Expected results (priors)

Based on literature:
- Per-asset Sharpe distribution: roughly 60% positive, 30% borderline, 10% noise
- Portfolio Sharpe: 0.9-1.1 after risk-parity weighting
- Max DD: 12-18%
- Correlation to S&P 500: ~0.1 (true diversification)
- Best months / worst months: typically slightly positive skew

## When to actually do this

Defer until Path 1 (gold + QQQ TSM in a futures account, paper-traded for
6 months) is complete and live performance has validated the simpler strategy.
There's no point building a 25-asset basket if the 2-asset basket doesn't live
up to its backtest.
