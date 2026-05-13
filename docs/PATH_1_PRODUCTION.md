# Path 1 — Productionize gold + QQQ TSM

**Status:** In progress. This is the chosen primary path.

**Premise:** The biggest profit improvement isn't a better Sharpe — it's better
capital structure. Sharpe 0.7 at 25% volatility makes 2.5× more dollars than
Sharpe 0.9 at 10% volatility. Futures unlock the volatility dial without
changing the strategy. The strategy stays simple; the *risk dial* does the work.

```
profit ≈ Sharpe × volatility × capital
```

Of the three multipliers, strategy work moves the first by ~20-50%; capital-
structure work moves the second by 3-10×.

---

## Step 1 — Switch from ETFs to futures

Replace each ETF in the validated strategy with its futures equivalent:

| ETF | Underlying | Std Futures | Micro Futures (recommended) |
|---|---|---|---|
| GLD | Gold | GC (100 oz) | **MGC (10 oz)** |
| QQQ | Nasdaq-100 | NQ ($20×index) | **MNQ ($2×index)** |
| SPY | S&P 500 | ES ($50×index) | **MES ($5×index)** |
| TLT | Long bonds | ZB ($1000×price) | (no micro) |

**Why micros are the retail-realistic choice:**
- MGC: ~$1,000 initial margin per contract (vs $10k for GC)
- MNQ: ~$2,000 initial margin per contract (vs $20k for NQ)
- Can run a balanced 2-contract position with $5-10k of margin
- Liquid enough during US market hours

**Benefits vs ETFs:**
- **Leverage**: 5-20× available; ETFs cap at 2× via Reg-T margin
- **Costs**: ~$1-3 per side per contract vs 1-5 bps on ETFs. For TSM with
  ~12 rebalances/year, this is roughly 5-10× lower
- **Taxes (US)**: Section 1256 60/40 treatment — 60% of gains taxed at LT
  rate regardless of holding period. Saves ~10% of after-tax returns vs
  short-term capital gains on ETF turnover
- **No expense ratio** drag (~0.4% on GLD, 0.20% on QQQ)
- **23-hour trading session** vs 6.5 hours for ETFs

**Verify in the backtest:** `run_tsm_futures.py` runs the validated TSM strategy
with proper contract multipliers, per-contract commissions, integer contract
sizing, and margin tracking. Outputs return/DD/wipeout at each leverage level.

## Step 2 — Paper trade for 6 months

The orchestrator already supports paper mode via `mode: paper` in the config.
What's needed for live wiring:

1. **Open a paper account** at a broker that supports futures (Interactive
   Brokers, NinjaTrader, Tradovate, AMP). All offer free paper trading.
2. **Wire TSM into orchestrator**:
   - New file `live/tsm_runner.py` that:
     - Reads daily closes at end-of-day
     - Computes the TSM signal & sizing
     - Submits orders only at month-start (rebalance trigger)
     - Logs every signal/order/fill to `logs/tsm_live.jsonl`
3. **Daily monitoring**:
   - Reconciliation report: live equity vs backtest equity replayed forward
   - Alert if live deviates from backtest by > 1 daily vol
4. **Don't tune anything during paper trading**. The whole point is to
   confirm the backtest replicates live. Tuning mid-validation invalidates
   the test.

**Pass criteria** (after 6 months):
- Live Sharpe within ±0.3 of backtest Sharpe
- Live max DD ≤ 1.5× backtest max DD
- No operational issues (missed signals, mis-sized orders, broker glitches)
- Subjectively trade-able — i.e., you didn't override the system because of a
  drawdown that "felt too painful"

## Step 3 — Scale capital

After paper validation, the question becomes: what volatility level matches
your real-world risk tolerance?

Reference points based on the validated strategy (gold + QQQ TSM portfolio,
Sharpe ~0.7 in backtest):

| Annualized vol | Expected return | Realistic max DD | Notes |
|---|---|---|---|
| 5% | 3.5% | -10% | Conservative; underperforms cash + S&P during bull markets |
| 10% | 7% | -22% | Backtest default; similar overall risk to 60/40 portfolio |
| 15% | 10.5% | -33% | Aggressive; comparable risk to all-equity portfolio |
| 25% | 17.5% | -55% | Pro-trader territory; expect 50% peak-to-trough at some point |
| 50% | 35% | "wipeout possible" | Don't do this with money you can't lose |

**Rule of thumb:** pick a volatility target you can stomach during a 2× max-DD
event (the historical max DD is a *floor* on future max DD, not a ceiling).
If 10% backtest DD is your psychological limit, run at the vol level where
backtest DD × 2 = 10% → roughly 5% vol target.

**Sizing the position:**
```
contract_count = round(target_vol × equity / (asset_vol × contract_multiplier × price))
```

For MGC at $2000/oz with 15% asset vol and a $50k account targeting 10% vol:
```
contracts = round(0.10 × 50000 / (0.15 × 10 × 2000)) = round(1.67) = 2 MGC
notional = 2 × 10 × 2000 = $40,000
margin used = 2 × $1,000 = $2,000  (4% of equity)
```

The strategy works at any account size from ~$10k (with micros) upward.

## Step 4 (optional) — Production hardening

Things worth doing if the paper trading validates:

- **Two-broker redundancy** — submit to primary, log to secondary for audit
- **Heartbeat alerts** — Telegram/Discord notification if no signal computed
  by 4:30pm UTC on a rebalance day
- **Pre-flight checks** before each rebalance: account margin, no fat-finger
  size, no halted contract
- **Kill switch** that closes all positions and disables trading on a
  configurable drawdown threshold (e.g., -15% from high-water mark)

## What's deliberately deferred

- Multi-asset CTA basket → see `PATH_2_CTA_BASKET.md`
- ML overlay → don't bolt ML onto a working strategy without need
- Alternative strategies (mean-reversion, options) → high overfit risk vs
  marginal expected improvement on top of a validated TSM result

## Decision points along the way

1. **After futures backtest**: leverage level chosen for paper trading
2. **After 3 months paper**: continue or abandon based on tracking error
3. **After 6 months paper**: go live with what % of intended capital
4. **After 6 months live**: scale up to full intended capital
