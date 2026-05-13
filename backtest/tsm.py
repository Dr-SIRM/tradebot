"""Time-series momentum (TSM) — pure simulation functions.

Reference: Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum".

The strategy at each rebalance:
  1. Look at the trailing `lookback_days` total return, excluding the most
     recent `skip_days` (the "12-1" month skip removes short-term reversal).
  2. Signal = sign of that return → long, short, or flat.
  3. Size the position to target a fixed annualized volatility by scaling
     1/realized_vol (computed over `vol_lookback_days` of daily returns).
  4. Hold until the next rebalance.

Why this design:
- Parameter-free in terms of fitting (the only knobs are lookback/skip/vol
  target — these come from the literature, not from in-sample optimization).
- Volatility-targeting equalizes risk across very different asset classes
  (so 4%-ATR BTC and 0.5%-ATR EURUSD get comparable position risk).
- Monthly rebalancing keeps turnover (and therefore costs) low.

The simulation is daily for P&L tracking but only changes positions on
rebalance dates. Costs are applied to turnover at each rebalance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TSMParams:
    lookback_days: int = 252       # ~12 months of trading days
    skip_days: int = 21            # ~1 month — drop the most recent to avoid reversal
    vol_lookback_days: int = 60    # ~3 months for realized vol estimate
    target_vol: float = 0.10       # 10% annualized portfolio vol target
    max_leverage: float = 2.0      # cap on |position| (avoid blowups in low-vol regimes)
    cost_bps_per_turnover: float = 10.0  # round-trip; applied to |Δposition|
    rebalance: str = "monthly"     # 'daily' | 'weekly' | 'monthly'
    initial_equity: float = 100_000.0


def _rebalance_mask(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """Boolean mask marking rebalance days. First trading day of period."""
    if freq == "daily":
        return pd.Series(True, index=index)
    if freq == "weekly":
        return pd.Series(index.to_series().dt.isocalendar().week
                          != index.to_series().shift(1).dt.isocalendar().week,
                          index=index).fillna(True)
    if freq == "monthly":
        return pd.Series(index.to_series().dt.month
                          != index.to_series().shift(1).dt.month,
                          index=index).fillna(True)
    raise ValueError(f"unknown rebalance freq: {freq!r}")


def tsm_signal(close: pd.Series, lookback_days: int, skip_days: int) -> pd.Series:
    """At each bar t, return the sign of cumulative log return from
    t-(lookback+skip) to t-skip. Returns -1, 0, or +1."""
    log_close = np.log(close)
    # Cumulative log return over `lookback_days` ending `skip_days` ago.
    # log_close.diff(lookback_days) gives total log return over lookback ending at t.
    # Then we shift forward by skip_days so the window ends `skip` bars ago.
    mom = log_close.diff(lookback_days).shift(skip_days)
    sig = np.sign(mom).fillna(0.0)
    return sig


def vol_scaled_weight(close: pd.Series, vol_lookback_days: int,
                       target_vol: float, max_leverage: float) -> pd.Series:
    """Scale factor so that signal × scale targets `target_vol` annualized."""
    rets = close.pct_change()
    realized_vol = rets.rolling(vol_lookback_days, min_periods=vol_lookback_days // 2) \
                       .std() * np.sqrt(252)
    weight = (target_vol / realized_vol).clip(upper=max_leverage)
    return weight.fillna(0.0)


def simulate_tsm(close: pd.Series, params: TSMParams) -> dict:
    """Simulate TSM on a single price series. Returns a dict with:
      - equity: pd.Series of equity curve (daily)
      - position: pd.Series of position weight (+1.0 = long full vol-target, etc.)
      - daily_returns: net daily portfolio returns (after costs)
      - monthly_returns: net returns aggregated per rebalance period
      - trades: list of dicts (one per position change) for MC bootstrap
      - turnover_total: total absolute turnover (sanity check)
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must have a DatetimeIndex")
    close = close.sort_index()

    signal = tsm_signal(close, params.lookback_days, params.skip_days)
    weight = vol_scaled_weight(close, params.vol_lookback_days,
                                params.target_vol, params.max_leverage)
    raw_position = signal * weight

    # Apply rebalance schedule: only change position on rebalance days.
    is_rebal = _rebalance_mask(close.index, params.rebalance)
    position = raw_position.where(is_rebal).ffill().fillna(0.0)

    # Warmup: zero out positions until we have enough history for the signal
    warmup = params.lookback_days + params.skip_days + params.vol_lookback_days
    if warmup < len(position):
        position.iloc[:warmup] = 0.0

    # Daily P&L: yesterday's position × today's return.
    rets = close.pct_change().fillna(0.0)
    gross_daily = position.shift(1).fillna(0.0) * rets

    # Costs on |Δposition|. cost_bps is per unit of turnover (round-trip basis).
    turnover = position.diff().abs().fillna(0.0)
    costs_daily = turnover * (params.cost_bps_per_turnover / 10000.0)

    net_daily = gross_daily - costs_daily
    equity = (1.0 + net_daily).cumprod() * params.initial_equity

    # Build "trade" records: each rebalance where position actually changed.
    trades: list[dict] = []
    changed = position.diff().fillna(position).abs() > 1e-9
    prev_pos = 0.0
    prev_time = position.index[0]
    prev_eq = params.initial_equity
    for ts, new_pos in position[changed].items():
        period_eq = float(equity.loc[ts])
        period_ret = (period_eq / prev_eq) - 1.0 if prev_eq > 0 else 0.0
        trades.append({
            "rebalance_time": ts,
            "prev_position": float(prev_pos),
            "new_position": float(new_pos),
            "turnover": abs(float(new_pos) - float(prev_pos)),
            "period_pct_return": float(period_ret),
        })
        prev_pos = float(new_pos)
        prev_time = ts
        prev_eq = period_eq

    monthly_returns = net_daily.resample("ME").apply(lambda s: (1 + s).prod() - 1).dropna()

    return {
        "equity": equity,
        "position": position,
        "daily_returns": net_daily,
        "gross_daily": gross_daily,
        "costs_daily": costs_daily,
        "monthly_returns": monthly_returns,
        "trades": trades,
        "turnover_total": float(turnover.sum()),
        "signal": signal,
        "weight": weight,
    }


def summarize(sim: dict, periods_per_year: int = 252) -> dict:
    """Compute headline stats from a simulate_tsm result."""
    rets = sim["daily_returns"]
    rets = rets[rets.index >= rets[rets != 0].index.min() if (rets != 0).any() else rets.index[0]]
    if len(rets) == 0:
        return {"sharpe": 0.0, "total_return": 0.0, "max_dd": 0.0,
                "n_trades": 0, "annualized_vol": 0.0}

    eq = sim["equity"].loc[rets.index]
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) > 1 else 0.0
    mu = rets.mean() * periods_per_year
    sigma = rets.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = float(mu / sigma) if sigma > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    return {
        "sharpe": sharpe,
        "annualized_return": float(mu),
        "annualized_vol": float(sigma),
        "total_return": total_return,
        "max_dd": float(dd),
        "n_trades": len(sim.get("trades", [])),
        "turnover_total": float(sim.get("turnover_total", 0.0)),
    }


def simulate_tsm_portfolio(
    prices: dict[str, pd.Series],
    params: TSMParams,
    target_portfolio_vol: float | None = 0.10,
    vol_lookback_days: int = 60,
) -> dict:
    """Equal-weight portfolio of per-asset TSM streams.

    Each asset runs through simulate_tsm() independently. At each date, the
    portfolio holds an equal weight in every 'live' asset (signal active, past
    warmup). When N_live varies (assets coming online at different dates,
    or going flat during weak-signal periods) the equal-weight allocation
    rebalances automatically.

    Per-asset streams are already vol-targeted, so the equal-weight portfolio
    has volatility roughly target_per_asset_vol / sqrt(N_live) when assets are
    uncorrelated. To hit a stable portfolio vol target across varying N, an
    optional rolling realized-vol rescale is applied on top (causal: vol
    estimated from past returns only, shifted forward one day).

    Args:
        prices: {symbol: close-price Series}, each with a DatetimeIndex.
        params: TSMParams used for every per-asset simulation.
        target_portfolio_vol: If set, rescale the daily portfolio return so
            its rolling realized vol matches this target. Pass None to skip.
        vol_lookback_days: Window for realized-vol estimate (if rescaling).

    Returns dict with:
        equity, daily_returns: portfolio-level series (DatetimeIndex)
        per_asset_sims: {symbol: simulate_tsm result} for each input asset
        n_live: int Series — number of live assets at each date
        portfolio_weights: DataFrame symbol×date of weights actually applied
        scale: pd.Series of the vol-target rescale factor (1.0 if disabled)
    """
    if not prices:
        raise ValueError("prices is empty")

    per_asset = {sym: simulate_tsm(close.sort_index(), params)
                 for sym, close in prices.items()}

    rets_df = pd.DataFrame({sym: per_asset[sym]["daily_returns"]
                             for sym in per_asset}).sort_index()
    pos_df = pd.DataFrame({sym: per_asset[sym]["position"]
                            for sym in per_asset}).reindex(rets_df.index)

    # 'Live' = strategy is actually holding (non-zero position). NaN from
    # asset-not-yet-in-data also counts as not-live.
    is_live = (pos_df.fillna(0.0) != 0.0)
    n_live = is_live.sum(axis=1).astype(int)

    # Equal weight across live assets. weights[t,sym] = 1/N_live(t) if live else 0.
    weights = is_live.astype(float).div(n_live.replace(0, np.nan), axis=0).fillna(0.0)

    # Per-asset return when not live should be 0 (no exposure)
    rets_masked = rets_df.where(is_live, 0.0).fillna(0.0)
    unscaled = (rets_masked * weights).sum(axis=1)

    # Optional realized-vol rescaling to hit a fixed portfolio target.
    # Causal: vol estimated from returns up to t-1 used to size for day t.
    if target_portfolio_vol is None:
        scale = pd.Series(1.0, index=unscaled.index)
    else:
        realized_vol = unscaled.rolling(vol_lookback_days,
                                          min_periods=vol_lookback_days // 2) \
                                .std().shift(1) * np.sqrt(252)
        scale = (target_portfolio_vol / realized_vol).clip(upper=params.max_leverage)
        scale = scale.fillna(0.0)

    portfolio_rets = scale * unscaled
    equity = (1.0 + portfolio_rets).cumprod() * params.initial_equity

    return {
        "equity": equity,
        "daily_returns": portfolio_rets,
        "unscaled_returns": unscaled,
        "scale": scale,
        "n_live": n_live,
        "portfolio_weights": weights,
        "per_asset_sims": per_asset,
    }


def summarize_portfolio(sim: dict, periods_per_year: int = 252) -> dict:
    """Headline stats for a simulate_tsm_portfolio result, including the
    per-asset Sharpes so the user can see what's pulling the portfolio
    around."""
    base = summarize(sim, periods_per_year=periods_per_year)
    per_asset = {}
    for sym, asset_sim in sim["per_asset_sims"].items():
        per_asset[sym] = summarize(asset_sim, periods_per_year=periods_per_year)
    base["per_asset"] = per_asset
    base["avg_n_live"] = float(sim["n_live"].mean()) if len(sim["n_live"]) else 0.0
    base["max_n_live"] = int(sim["n_live"].max()) if len(sim["n_live"]) else 0
    return base
