# Multi-Asset Algorithmic Trading Bot

Production-grade, asset-agnostic algorithmic trading framework. Trades crypto, equities, forex, and commodities through a unified broker abstraction layer with strict risk management, multiple strategies, regime detection, and full backtesting infrastructure.

## ⚠️ Important disclaimers — read first

1. **Paper trading by default.** Live mode requires explicit env-var opt-in (`TRADEBOT_LIVE=1`) AND valid broker credentials. Don't switch to live until you've run paper for 60+ days across varied market conditions.
2. **No backtest is ground truth.** Walk-forward and Monte Carlo modules exist to keep you honest. Treat any in-sample Sharpe > 2 as suspicious.
3. **Broker adapters are written against documented APIs but require testing in each broker's sandbox before trusting live capital.**
4. **The ML model is a scoring layer, not a strategy.** It needs to be trained on your data — `ml/train.py` does that.
5. **Past performance does not predict future results.** Most retail algo strategies lose money. Plan accordingly.

## Architecture

```
tradebot/
├── config/              # YAML configs — all parameters live here, no hardcoding
│   └── default.yaml
├── data/                # Market data feeds and indicator calculation
│   ├── feed.py          # Async OHLCV feed adapter (multi-broker)
│   ├── indicators.py    # EMA/RSI/ATR/ADX/BB/volume — vectorized
│   └── calendar.py      # Economic calendar blackout windows
├── strategy/            # Trading strategies + regime selector
│   ├── base.py          # Abstract Strategy interface
│   ├── momentum.py      # EMA cross + RSI + volume surge
│   ├── mean_reversion.py# BB squeeze + RSI extremes
│   ├── breakout.py      # S/R + volume confirmation
│   ├── regime.py        # ADX/ATR-based regime classifier
│   └── selector.py      # Picks active strategy per asset per bar
├── risk/                # All risk gates
│   ├── manager.py       # Account-level P&L, drawdown, trade limits
│   ├── position_sizer.py# 1% risk + ATR stops + conviction scaling
│   └── correlation.py   # Cross-asset exposure check
├── execution/           # Broker abstraction + order routing
│   ├── broker_base.py   # Abstract broker interface
│   ├── alpaca.py        # Equities/ETFs
│   ├── binance.py       # Crypto (spot + perp)
│   ├── oanda.py         # Forex
│   ├── paper.py         # In-memory paper broker
│   └── router.py        # Smart limit-first routing + fill tracking
├── ml/                  # XGBoost trade-setup scorer
│   ├── features.py      # Feature engineering pipeline
│   ├── model.py         # Wrapper, load/save, predict
│   └── train.py         # Training script
├── backtest/            # Validation harness
│   ├── engine.py        # Vectorized bar-by-bar simulator
│   ├── metrics.py       # Sharpe/Sortino/PF/maxDD/etc.
│   ├── walkforward.py   # Rolling out-of-sample windows
│   ├── montecarlo.py    # Trade-bootstrap stress test
│   └── sensitivity.py   # Parameter grid sensitivity scan
├── alerts/              # Telegram/Discord notifications
│   └── notifier.py
├── dashboard/           # Live terminal UI
│   └── tui.py
├── utils/               # Logging, types, helpers
│   ├── logging.py
│   ├── types.py
│   └── time.py
├── tests/               # Unit tests for indicators, risk, regime
├── orchestrator.py      # Main async event loop
├── run_live.py          # Entry point: live/paper trading
├── run_backtest.py      # Entry point: backtests
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1) Backtest a single strategy combo on historical data
python run_backtest.py --data data/btc_5m.csv --symbol BTC/USDT --asset-class crypto

# 2) Walk-forward validation (rolling out-of-sample windows)
python run_backtest.py --data data/btc_5m.csv --symbol BTC/USDT --asset-class crypto --walkforward

# 3) Monte Carlo on the in-sample trade returns
python run_backtest.py --data data/btc_5m.csv --symbol BTC/USDT --asset-class crypto --monte-carlo --mc-runs 1000

# 4) Parameter sensitivity (overfit detector)
python run_backtest.py --data data/btc_5m.csv --symbol BTC/USDT --asset-class crypto --sensitivity

# 5) Or all at once
python run_backtest.py --data data/btc_5m.csv --symbol BTC/USDT --asset-class crypto \
    --walkforward --monte-carlo --sensitivity

# 6) Paper trade (default, safe)
python run_live.py --config config/default.yaml

# 7) LIVE trade — only after extensive paper validation
TRADEBOT_LIVE=1 python run_live.py --config config/default.yaml
```

CSV format: columns `timestamp,open,high,low,close,volume` (timestamp parseable
by pandas). The HTF dataframe is auto-resampled from the LTF if you don't
pass `--data-htf` explicitly.

## Recommended validation workflow

1. **Backtest** in-sample → look for Sharpe, profit factor, expectancy_R that are positive AND realistic (>2 Sharpe in-sample is usually a red flag).
2. **Walk-forward** → out-of-sample Sharpe should be within 50% of in-sample. If it collapses, you're overfit.
3. **Monte Carlo** → P(profitable) should be >60%, max-DD p95 should be within your tolerance.
4. **Sensitivity** → small parameter changes should produce small metric changes. Stability score (mean Sharpe / std Sharpe) >2 is good.
5. **Paper trade for 60+ days** across varied market conditions before considering live.
6. **Start live with 10% of intended capital.** Scale up only after weeks of matching paper-trade behaviour.

## Risk controls (enforced, not advisory)

| Control                  | Default      | Where enforced                          |
|--------------------------|--------------|-----------------------------------------|
| Max risk per trade       | 1% equity    | `risk/position_sizer.py`                |
| Min reward:risk          | 2.5:1        | `risk/manager.py::approve_trade`        |
| ATR stop multiplier      | 1.5–2.0×     | `risk/position_sizer.py`                |
| Daily drawdown halt      | 4%           | `risk/manager.py::update_equity_mark`   |
| Weekly drawdown halt     | 8%           | `risk/manager.py::update_equity_mark`   |
| Max concurrent positions | 3            | `risk/manager.py::approve_trade`        |
| Correlation cap          | 1 per cluster| `risk/correlation.py`                   |
| News blackout            | ±15 min      | `data/calendar.py`                      |
| Trailing stop            | activates +1.5R | `risk/manager.py::update_trailing`   |

## Modes

- **Backtest** — historical CSV/parquet replay with full metrics suite
- **Paper** — live data, simulated fills, identical code path to live
- **Live** — real broker orders; requires `TRADEBOT_LIVE=1`

## License

Use at your own risk. No warranty. You alone are responsible for any losses.
