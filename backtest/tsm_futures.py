"""TSM simulation with realistic futures cost and margin model.

This is the same TSM signal logic as `backtest/tsm.py` but with three changes
that match real-world futures trading:

  1. **Integer contract sizing.** Real positions are in whole contracts. Small
     accounts therefore have rounding-error granularity (e.g., a 0.4-contract
     ideal position rounds to 0 → underexposed).
  2. **Per-contract commission**, not basis-points of notional. Round-trip
     costs are ~$1-5 per contract; for our turnover this is roughly 10x lower
     than the ETF bps model.
  3. **Margin tracking with wipeout detection.** Each day, required margin =
     |n_contracts| × initial_margin_per_contract. If that exceeds equity, the
     position would be force-liquidated in real life. We flag those days and
     terminate the simulation if a wipeout happens — that's the truth about
     leverage limits.

The CORE strategy logic (signal, vol-targeting, rebalance) is unchanged —
we're just modeling the *execution* properly. The point is to show what
leverage levels are actually achievable before the strategy blows up on
historical data.

Key result we want to expose: at the chosen vol-target × leverage, the worst
historical drawdown must stay below the equity. Otherwise the strategy would
have been liquidated and the headline Sharpe is a fiction.

Returns are computed in pct-of-equity space (which is scale-invariant), so
the precise futures price level doesn't matter for P&L — only for contract
sizing and margin. A small approximation (using the ETF proxy price for
contract notional rather than the actual futures price) introduces only
granularity error at very small account sizes; the strategy result is
correct in expectation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.tsm import (
    TSMParams,
    _rebalance_mask,
    multi_horizon_signal,
    tsm_signal,
    vol_scaled_weight,
)


@dataclass
class FuturesContract:
    symbol: str
    multiplier: float          # contract notional = multiplier × futures_price
    initial_margin: float      # $ per contract held overnight
    commission_per_side: float = 2.50
    tick_size: float = 0.01
    tick_value: float = 0.0    # unused in P&L (we work in returns) but useful for docs
    name: str = ""
    # Approximate ratio: futures_price / proxy_etf_price. Only affects contract
    # sizing & margin; the P&L is scale-invariant. Defaults to 1.0 if the proxy
    # IS the underlying (no scaling needed); set to 10x for GLD->gold, etc.
    proxy_price_scale: float = 1.0


@dataclass
class FuturesSimResult:
    equity: pd.Series                 # daily equity curve
    daily_returns: pd.Series          # net daily portfolio return (% of equity)
    contracts: pd.Series              # integer contracts held each day
    margin_used: pd.Series            # $ margin required each day
    margin_pct: pd.Series             # margin / equity each day
    commissions_paid: float
    margin_breaches: int              # days where margin > equity
    wiped_out: bool                   # equity hit zero or went negative
    wipeout_date: pd.Timestamp | None
    raw_position_pct: pd.Series       # signal × vol_target × leverage (before rounding)
    contract: FuturesContract
    leverage: float
    derisk_flag: pd.Series | None = None  # bool series of derisked days (None when off)
    derisk_active_days: int = 0


def simulate_tsm_futures(
    close: pd.Series,
    contract: FuturesContract,
    params: TSMParams,
    leverage: float = 1.0,
    initial_equity: float = 100_000.0,
    halt_on_wipeout: bool = True,
) -> FuturesSimResult:
    """Run TSM with a futures cost & margin model. See module docstring for
    the modeling assumptions.

    Args:
        close: ETF proxy daily close (e.g., GLD prices for gold). The signal
            and returns are computed from this series; futures-level prices
            are inferred by multiplying with contract.proxy_price_scale.
        contract: FuturesContract specs.
        params: TSMParams. target_vol is the *per-asset* annualized vol target;
            leverage stacks on top.
        leverage: Scales the position size beyond what vol-targeting alone
            would dictate. leverage=1 means run at params.target_vol;
            leverage=2 means 2× position size (and 2× vol).
        initial_equity: Starting account equity in $.
        halt_on_wipeout: If True, freeze the equity series once equity ≤ 0.

    Returns: FuturesSimResult with equity curve, contract history, margin
    usage series, and wipeout diagnostics.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must have a DatetimeIndex")
    close = close.sort_index()
    n = len(close)
    effective_lookback = (max(params.multi_horizon_lookbacks)
                            if params.multi_horizon_lookbacks
                            else params.lookback_days)
    if n < effective_lookback + params.skip_days + params.vol_lookback_days + 10:
        raise ValueError(
            f"need >= {effective_lookback + params.skip_days + params.vol_lookback_days + 10} "
            f"bars; got {n}"
        )

    # ---- Same TSM signal & vol-scaled weight as the ETF simulator ----
    if params.multi_horizon_lookbacks:
        signal = multi_horizon_signal(close, params.multi_horizon_lookbacks,
                                        params.skip_days)
    else:
        signal = tsm_signal(close, params.lookback_days, params.skip_days)
    # Effective vol target is the strategy's vol times the leverage factor.
    vol_target_effective = params.target_vol * leverage
    # max_leverage cap: the vol-scaling can't multiply position beyond this;
    # we scale that up too so leverage=2 doesn't get clipped by a 2x cap.
    max_lev_effective = params.max_leverage * leverage
    weight = vol_scaled_weight(close, params.vol_lookback_days,
                                vol_target_effective, max_lev_effective)
    raw_position_pct = (signal * weight).reindex(close.index).fillna(0.0)

    # Rebalance gating
    is_rebal = _rebalance_mask(close.index, params.rebalance)
    position_pct = raw_position_pct.where(is_rebal).ffill().fillna(0.0)
    warmup = effective_lookback + params.skip_days + params.vol_lookback_days
    if warmup < len(position_pct):
        position_pct.iloc[:warmup] = 0.0

    # ---- Day-by-day simulation with integer contracts and margin tracking ----
    rets = close.pct_change().fillna(0.0).values
    pos_pct = position_pct.values
    futures_prices = close.values * contract.proxy_price_scale

    equity = np.zeros(n)
    contracts_arr = np.zeros(n, dtype=np.int64)
    margin_used = np.zeros(n)
    daily_rets = np.zeros(n)

    equity[0] = initial_equity
    prev_contracts = 0
    total_commission = 0.0
    margin_breaches = 0
    wipeout_idx: int | None = None

    # Drawdown-based de-risking state.
    derisk_enabled = params.derisk_dd_threshold > 0
    derisk_flag_arr = np.zeros(n, dtype=bool)
    peak_equity = initial_equity
    is_derisked = False

    for i in range(n):
        # Determine target contracts at start of bar i based on prior equity
        # and the position fraction decided at the most recent rebalance.
        prev_eq = equity[i - 1] if i > 0 else initial_equity
        if prev_eq <= 0:
            # Already wiped out — freeze everything
            equity[i] = 0.0
            contracts_arr[i] = 0
            margin_used[i] = 0.0
            daily_rets[i] = 0.0
            continue

        # DD-derisk: state transitions happen based on prev_eq (info available
        # at the start of bar i). flag[i] records the factor APPLIED to bar i.
        if derisk_enabled:
            if prev_eq > peak_equity:
                peak_equity = prev_eq
            dd_now = (prev_eq - peak_equity) / peak_equity if peak_equity > 0 else 0.0
            # Transition first (using state of prior bar), then record flag.
            if not is_derisked and dd_now <= -params.derisk_dd_threshold:
                is_derisked = True
            elif is_derisked and dd_now >= -params.derisk_recovery_threshold:
                is_derisked = False
        derisk_flag_arr[i] = is_derisked
        derisk_factor = params.derisk_scale if is_derisked else 1.0

        target_dollars = pos_pct[i] * prev_eq * derisk_factor   # signed $ exposure
        contract_notional = contract.multiplier * futures_prices[i]
        if contract_notional <= 0:
            n_contracts = 0
        else:
            n_contracts = int(np.round(target_dollars / contract_notional))

        # Apply commission on the *change* in contracts (per-side count)
        d_contracts = abs(n_contracts - prev_contracts)
        commission_today = d_contracts * contract.commission_per_side
        total_commission += commission_today

        # P&L = previous contracts × multiplier × price change
        if i > 0:
            price_chg = futures_prices[i] - futures_prices[i - 1]
            pnl = prev_contracts * contract.multiplier * price_chg
        else:
            pnl = 0.0

        equity[i] = prev_eq + pnl - commission_today
        daily_rets[i] = (equity[i] - prev_eq) / prev_eq if prev_eq > 0 else 0.0

        margin_req = abs(n_contracts) * contract.initial_margin
        margin_used[i] = margin_req
        if margin_req > equity[i]:
            margin_breaches += 1
        contracts_arr[i] = n_contracts

        # Wipeout check — equity ≤ 0 means force-liquidated in real life
        if equity[i] <= 0:
            if wipeout_idx is None:
                wipeout_idx = i
            if halt_on_wipeout:
                # Zero out future bars; leave loop running so output is full-length
                equity[i] = 0.0

        prev_contracts = n_contracts

    eq_series = pd.Series(equity, index=close.index)
    ret_series = pd.Series(daily_rets, index=close.index)
    contracts_series = pd.Series(contracts_arr, index=close.index)
    margin_series = pd.Series(margin_used, index=close.index)
    # avoid 0/0 → NaN: equity is always > 0 until wipeout, in which case we want inf
    with np.errstate(divide="ignore", invalid="ignore"):
        margin_pct = pd.Series(np.where(equity > 0, margin_used / equity, np.inf),
                                index=close.index)

    derisk_series = (pd.Series(derisk_flag_arr, index=close.index)
                      if derisk_enabled else None)
    return FuturesSimResult(
        equity=eq_series,
        daily_returns=ret_series,
        contracts=contracts_series,
        margin_used=margin_series,
        margin_pct=margin_pct,
        commissions_paid=total_commission,
        margin_breaches=margin_breaches,
        wiped_out=wipeout_idx is not None,
        wipeout_date=close.index[wipeout_idx] if wipeout_idx is not None else None,
        raw_position_pct=position_pct,
        contract=contract,
        leverage=leverage,
        derisk_flag=derisk_series,
        derisk_active_days=int(derisk_flag_arr.sum()) if derisk_enabled else 0,
    )


def summarize_futures(res: FuturesSimResult,
                       periods_per_year: int = 252) -> dict:
    """Headline metrics for a futures simulation."""
    rets = res.daily_returns
    # Trim to the live region (after warmup, before wipeout)
    nonzero_idx = rets[rets != 0].index
    if len(nonzero_idx) == 0:
        return {"sharpe": 0.0, "annualized_return": 0.0, "annualized_vol": 0.0,
                "max_dd": 0.0, "total_return": -1.0 if res.wiped_out else 0.0,
                "wiped_out": res.wiped_out, "wipeout_date": res.wipeout_date,
                "commissions_paid": float(res.commissions_paid),
                "margin_breaches": int(res.margin_breaches),
                "max_margin_pct": float(res.margin_pct.replace(np.inf, np.nan).max()
                                          if len(res.margin_pct) else 0.0),
                "leverage": float(res.leverage),
                "contract": res.contract.symbol}

    live_rets = rets.loc[nonzero_idx[0]:]
    eq = res.equity.loc[nonzero_idx[0]:]
    mu = live_rets.mean() * periods_per_year
    sigma = live_rets.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = float(mu / sigma) if sigma > 0 else 0.0
    # Total return: if wiped out, -100%; else final/initial - 1
    initial = eq.iloc[0] if eq.iloc[0] > 0 else 1.0
    if res.wiped_out:
        total_return = -1.0
    else:
        total_return = float(eq.iloc[-1] / initial - 1.0)
    # Max drawdown
    running_max = eq.cummax()
    dd = ((eq - running_max) / running_max).min()

    return {
        "sharpe": sharpe,
        "annualized_return": float(mu),
        "annualized_vol": float(sigma),
        "max_dd": float(dd),
        "total_return": total_return,
        "wiped_out": res.wiped_out,
        "wipeout_date": res.wipeout_date,
        "commissions_paid": float(res.commissions_paid),
        "margin_breaches": int(res.margin_breaches),
        "max_margin_pct": float(res.margin_pct.replace(np.inf, np.nan).max()),
        "leverage": float(res.leverage),
        "contract": res.contract.symbol,
    }
