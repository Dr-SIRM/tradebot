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
    # Multi-horizon momentum: if set to a non-empty list, the signal is the
    # average TSM sign across these lookbacks (instead of the single
    # `lookback_days`). Defends against PBO-style lookback fragility and
    # gives a continuous-magnitude signal. Common choice: [63, 126, 252]
    # (3m / 6m / 12m). When None, falls back to single-horizon at lookback_days.
    multi_horizon_lookbacks: list[int] | None = None
    # Drawdown-based de-risking: when running equity drawdown exceeds
    # `derisk_dd_threshold`, multiply subsequent positions by `derisk_scale`.
    # Restore full size when DD recovers to under `derisk_recovery_threshold`.
    # Hysteresis (the gap between the two thresholds) avoids whipsaw at the
    # boundary. Set derisk_dd_threshold=0 to disable.
    #
    # Doesn't typically improve Sharpe — the strategy also shrinks just before
    # the recovery rally — but materially cuts max DD. Trade-off: ~5-15% lower
    # max DD for ~0-5% lower Sharpe. Worth it for trade-ability.
    derisk_dd_threshold: float = 0.0       # 0 = off; typical 0.10-0.15
    derisk_scale: float = 0.5              # multiply positions by this when triggered
    derisk_recovery_threshold: float = 0.05  # restore full size below this DD


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


def apply_dd_derisk(
    returns: pd.Series,
    threshold: float,
    scale: float,
    recovery_threshold: float,
    initial_equity: float = 1.0,
) -> tuple[pd.Series, pd.Series]:
    """Apply drawdown-based position scaling to a daily return series.

    Iterates daily, tracking the running equity peak and current drawdown.
    When DD exceeds `threshold`, subsequent daily returns are multiplied by
    `scale`. When DD recovers to below `recovery_threshold` (note the
    asymmetry — that's the hysteresis), full size is restored.

    Scaling positions by `f` is mathematically equivalent to scaling daily
    returns by `f` (since daily_return = position × asset_return). This
    function operates directly on the return series.

    Args:
        returns: daily return series (e.g. portfolio's pct returns or a
            single-asset TSM return stream).
        threshold: positive number. When equity DD ≤ -threshold, derisk.
        scale: factor applied while derisked (e.g. 0.5).
        recovery_threshold: positive number. When DD ≥ -recovery_threshold,
            restore full size. Must be < threshold for hysteresis.
        initial_equity: only affects the equity scale used internally; the
            derisk decision is invariant to it.

    Returns:
        (scaled_returns, derisk_flag) — both indexed like `returns`. The
        flag is True on days the derisk was active for that day's return.
    """
    if threshold <= 0:
        return returns.copy(), pd.Series(False, index=returns.index)
    if not (0.0 < recovery_threshold < threshold):
        raise ValueError(
            f"recovery_threshold ({recovery_threshold}) must be in "
            f"(0, threshold={threshold})"
        )

    rets_arr = returns.values
    n = len(rets_arr)
    out = np.zeros(n)
    flag = np.zeros(n, dtype=bool)
    eq = initial_equity
    peak = initial_equity
    is_derisked = False
    for i in range(n):
        # flag[i] records the factor APPLIED to bar i's return.
        flag[i] = is_derisked
        factor = scale if is_derisked else 1.0
        out[i] = rets_arr[i] * factor
        eq *= (1.0 + out[i])
        if eq > peak:
            peak = eq
        # dd is in [-1, 0]
        dd = (eq - peak) / peak if peak > 0 else 0.0
        # Transition state for the NEXT bar.
        if not is_derisked and dd <= -threshold:
            is_derisked = True
        elif is_derisked and dd >= -recovery_threshold:
            is_derisked = False
    return (pd.Series(out, index=returns.index),
            pd.Series(flag, index=returns.index))


def multi_horizon_signal(
    close: pd.Series,
    lookback_days_list: list[int],
    skip_days: int,
) -> pd.Series:
    """Average TSM signal across multiple lookback horizons.

    Equivalent to the Asness/Moskowitz multi-horizon momentum factor: at each
    bar, compute the binary momentum signal at each lookback, then average. The
    result is in [-1, +1] and acts as a *confidence* indicator — fully agreeing
    across horizons gives ±1, mixed agreement gives a smaller magnitude.

    Why this helps:
      - Single-lookback TSM is fragile to which exact lookback you pick (the
        PBO test showed lookback selection doesn't generalize).
      - Averaging several plausible horizons (e.g., 3m / 6m / 12m) is the
        textbook defense against this fragility.
      - The continuous signal also gives natural position-size scaling: half
        the horizons agreeing = half the position.
    """
    if not lookback_days_list:
        raise ValueError("lookback_days_list must not be empty")
    signals = [tsm_signal(close, lb, skip_days) for lb in lookback_days_list]
    return sum(signals) / len(signals)


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

    if params.multi_horizon_lookbacks:
        signal = multi_horizon_signal(close, params.multi_horizon_lookbacks,
                                        params.skip_days)
    else:
        signal = tsm_signal(close, params.lookback_days, params.skip_days)
    weight = vol_scaled_weight(close, params.vol_lookback_days,
                                params.target_vol, params.max_leverage)
    raw_position = signal * weight

    # Apply rebalance schedule: only change position on rebalance days.
    is_rebal = _rebalance_mask(close.index, params.rebalance)
    position = raw_position.where(is_rebal).ffill().fillna(0.0)

    # Warmup: zero out positions until we have enough history for the signal.
    # In multi-horizon mode the longest lookback drives the warmup.
    effective_lookback = (max(params.multi_horizon_lookbacks)
                            if params.multi_horizon_lookbacks
                            else params.lookback_days)
    warmup = effective_lookback + params.skip_days + params.vol_lookback_days
    if warmup < len(position):
        position.iloc[:warmup] = 0.0

    # Daily P&L: yesterday's position × today's return.
    rets = close.pct_change().fillna(0.0)
    gross_daily = position.shift(1).fillna(0.0) * rets

    # Costs on |Δposition|. cost_bps is per unit of turnover (round-trip basis).
    turnover = position.diff().abs().fillna(0.0)
    costs_daily = turnover * (params.cost_bps_per_turnover / 10000.0)

    net_daily = gross_daily - costs_daily

    # Drawdown-based de-risking (optional). Scales daily returns down by
    # `derisk_scale` once DD exceeds `derisk_dd_threshold`, until recovery.
    # Scaling returns is mathematically equivalent to scaling positions
    # since both gross and costs are linear in position size.
    if params.derisk_dd_threshold > 0:
        net_daily, derisk_flag = apply_dd_derisk(
            net_daily,
            threshold=params.derisk_dd_threshold,
            scale=params.derisk_scale,
            recovery_threshold=params.derisk_recovery_threshold,
            initial_equity=params.initial_equity,
        )
    else:
        derisk_flag = pd.Series(False, index=net_daily.index)

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
        "derisk_flag": derisk_flag,
        "derisk_active_days": int(derisk_flag.sum()),
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


def _portfolio_weights(
    rets_df: pd.DataFrame,
    is_live: pd.DataFrame,
    n_live: pd.Series,
    weighting: str,
    weighting_lookback_days: int,
) -> pd.DataFrame:
    """Per-asset portfolio weights at each date for a chosen weighting scheme.

    Output: T × N DataFrame where weights sum to 1 across columns wherever
    n_live > 0 (else all-zero row). Always shifted forward one day to be
    causal w.r.t. the daily return that will be applied next.
    """
    n_live_safe = n_live.replace(0, np.nan)

    if weighting == "equal":
        weights = is_live.astype(float).div(n_live_safe, axis=0).fillna(0.0)
        return weights  # equal-weight is naturally causal (no lookback used)

    if weighting == "inverse_vol":
        # Trailing realized vol of each asset's TSM return stream.
        rv = rets_df.rolling(weighting_lookback_days,
                               min_periods=weighting_lookback_days // 2).std()
        # Invert; mask not-live and assets with no/zero vol.
        inv = 1.0 / rv.where(rv > 0, np.nan)
        inv = inv.where(is_live, 0.0).fillna(0.0)
        # Normalize so rows sum to 1 where any asset has weight.
        row_sum = inv.sum(axis=1).replace(0, np.nan)
        weights = inv.div(row_sum, axis=0).fillna(0.0)
        # Causal: weights at t are used for the return at t+1, but the weights
        # themselves are computed on rolling windows ending at t, which
        # technically uses today's return. Shift forward by one day so weight
        # used at day t is computed only from data through t-1.
        return weights.shift(1).fillna(0.0)

    if weighting == "sharpe":
        # Trailing Sharpe of each asset's TSM stream. Negative Sharpes get 0.
        mu = rets_df.rolling(weighting_lookback_days,
                              min_periods=weighting_lookback_days // 2).mean()
        sd = rets_df.rolling(weighting_lookback_days,
                              min_periods=weighting_lookback_days // 2).std()
        sr = mu / sd.where(sd > 0, np.nan)
        sr = sr.clip(lower=0.0)              # negative-Sharpe → 0 weight
        sr = sr.where(is_live, 0.0).fillna(0.0)
        row_sum = sr.sum(axis=1).replace(0, np.nan)
        weights = sr.div(row_sum, axis=0).fillna(0.0)
        # If row sum is zero (everyone is in drawdown), fall back to equal
        # weight among live assets so we don't go fully flat — keeps the
        # strategy engaged when conditions normalize.
        equal_fallback = is_live.astype(float).div(n_live_safe, axis=0).fillna(0.0)
        weights = weights.where(weights.sum(axis=1) > 0, equal_fallback)
        return weights.shift(1).fillna(0.0)

    raise ValueError(f"unreachable: unknown weighting {weighting!r}")


def simulate_tsm_portfolio(
    prices: dict[str, pd.Series],
    params: TSMParams,
    target_portfolio_vol: float | None = 0.10,
    vol_lookback_days: int = 60,
    weighting: str = "equal",
    weighting_lookback_days: int = 252,
) -> dict:
    """Multi-asset TSM portfolio with selectable weighting scheme.

    Each asset runs through simulate_tsm() independently. At each date, the
    portfolio combines the live assets according to `weighting`:

      - "equal"        — same weight per live asset (the original behavior).
      - "inverse_vol"  — weight ∝ 1 / trailing-realized-vol of TSM return
                          stream. More stable streams get more allocation.
                          Since per-asset returns are already vol-targeted to
                          params.target_vol, this is approximately equal but
                          favors assets whose realized vol matches its target.
      - "sharpe"       — weight ∝ max(trailing realized Sharpe, 0). Negative-
                          Sharpe assets get zero allocation. Auto-drops losers
                          like ETH or HYG that drag down equal-weight portfolios.

    Weight estimates use a *trailing* window (`weighting_lookback_days`) and
    are shifted forward one day so they're causal — at each rebalance the
    weight is based only on observations strictly before the rebalance day.

    When `target_portfolio_vol` is set (default 10%), an additional rolling
    realized-vol rescale brings the combined return to that target. This is
    also causal and capped by params.max_leverage.

    Args:
        prices: {symbol: close-price Series}, each with a DatetimeIndex.
        params: TSMParams used for every per-asset simulation.
        target_portfolio_vol: Portfolio-level vol target after combining streams.
            None disables the rescale.
        vol_lookback_days: Window for the *portfolio-level* realized-vol rescale.
        weighting: "equal" | "inverse_vol" | "sharpe" — see above.
        weighting_lookback_days: Trailing window for inverse_vol / sharpe weight
            estimates. Defaults to 252 (~12 months) so the weighting is stable
            and doesn't whip around with short-term performance.

    Returns dict with:
        equity, daily_returns: portfolio-level series
        per_asset_sims: {symbol: simulate_tsm result}
        n_live: int Series — number of live assets at each date
        portfolio_weights: DataFrame symbol×date of weights actually applied
        scale: pd.Series of the vol-target rescale factor (1.0 if disabled)
        weighting: the scheme used (for the record)
    """
    if not prices:
        raise ValueError("prices is empty")
    if weighting not in {"equal", "inverse_vol", "sharpe"}:
        raise ValueError(f"unknown weighting {weighting!r}; "
                          f"use 'equal' | 'inverse_vol' | 'sharpe'")

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

    weights = _portfolio_weights(rets_df, is_live, n_live,
                                   weighting, weighting_lookback_days)

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

    # Portfolio-level drawdown de-risking: when the combined equity hits a
    # threshold drawdown, scale all subsequent daily returns down by
    # params.derisk_scale until recovery. Applied at the portfolio level
    # (not per-asset), which is the standard CTA practice.
    if params.derisk_dd_threshold > 0:
        portfolio_rets, derisk_flag = apply_dd_derisk(
            portfolio_rets,
            threshold=params.derisk_dd_threshold,
            scale=params.derisk_scale,
            recovery_threshold=params.derisk_recovery_threshold,
            initial_equity=params.initial_equity,
        )
    else:
        derisk_flag = pd.Series(False, index=portfolio_rets.index)

    equity = (1.0 + portfolio_rets).cumprod() * params.initial_equity

    return {
        "equity": equity,
        "daily_returns": portfolio_rets,
        "unscaled_returns": unscaled,
        "scale": scale,
        "n_live": n_live,
        "portfolio_weights": weights,
        "per_asset_sims": per_asset,
        "weighting": weighting,
        "derisk_flag": derisk_flag,
        "derisk_active_days": int(derisk_flag.sum()),
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
