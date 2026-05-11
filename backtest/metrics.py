"""Performance metrics for a list of Trades and an equity curve."""
from __future__ import annotations
from datetime import timedelta
import numpy as np
import pandas as pd

from utils.types import Trade


def equity_curve_from_trades(trades: list[Trade], starting_equity: float) -> pd.Series:
    """Step-function equity curve: equity stamps at each trade exit time."""
    if not trades:
        return pd.Series([starting_equity], index=[pd.Timestamp.utcnow()])
    sorted_t = sorted(trades, key=lambda t: t.exit_time)
    eq = starting_equity
    rows = [(sorted_t[0].entry_time, starting_equity)]
    for t in sorted_t:
        eq += t.pnl
        rows.append((t.exit_time, eq))
    idx, vals = zip(*rows)
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")


def sharpe(returns: pd.Series, periods_per_year: float = 252) -> float:
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=0))


def sortino(returns: pd.Series, periods_per_year: float = 252) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty or downside.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / downside.std(ddof=0))


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Return (max_dd_pct, max_dd_$). Both as positive numbers."""
    if equity.empty:
        return 0.0, 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak
    dd_dollars = peak - equity
    return float(dd.max()), float(dd_dollars.max())


def profit_factor(trades: list[Trade]) -> float:
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = sum(-t.pnl for t in trades if t.pnl < 0)
    if gross_loss <= 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def avg_trade_duration(trades: list[Trade]) -> timedelta:
    if not trades:
        return timedelta(0)
    durs = [(t.exit_time - t.entry_time) for t in trades]
    return sum(durs, timedelta(0)) / len(trades)


def expectancy_R(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t.r_multiple for t in trades]))


def compute_metrics(trades: list[Trade], starting_equity: float,
                    periods_per_year: float = 252) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "sharpe": 0.0,
                "sortino": 0.0, "max_drawdown_pct": 0.0, "max_drawdown_$": 0.0,
                "total_return_pct": 0.0, "expectancy_R": 0.0,
                "avg_trade_duration_hours": 0.0}

    eq = equity_curve_from_trades(trades, starting_equity)
    # Daily returns from equity curve (resample to daily for Sharpe)
    eq_daily = eq.resample("1D").last().ffill()
    rets = eq_daily.pct_change().dropna()

    wins = [t for t in trades if t.pnl > 0]
    total_return = (eq.iloc[-1] - starting_equity) / starting_equity
    dd_pct, dd_dollars = max_drawdown(eq_daily)
    avg_dur = avg_trade_duration(trades)

    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "profit_factor": profit_factor(trades),
        "sharpe": sharpe(rets, periods_per_year),
        "sortino": sortino(rets, periods_per_year),
        "max_drawdown_pct": dd_pct,
        "max_drawdown_$": dd_dollars,
        "total_return_pct": float(total_return),
        "expectancy_R": expectancy_R(trades),
        "avg_trade_duration_hours": avg_dur.total_seconds() / 3600,
        "final_equity": float(eq.iloc[-1]),
    }


def summary_str(m: dict) -> str:
    lines = [
        f"  trades:           {m['trades']}",
        f"  win rate:         {m['win_rate']:.2%}",
        f"  profit factor:    {m['profit_factor']:.2f}",
        f"  sharpe (daily):   {m['sharpe']:.2f}",
        f"  sortino (daily):  {m['sortino']:.2f}",
        f"  expectancy:       {m['expectancy_R']:.2f} R",
        f"  total return:     {m['total_return_pct']:.2%}",
        f"  max drawdown:     {m['max_drawdown_pct']:.2%} (${m['max_drawdown_$']:,.0f})",
        f"  avg duration:     {m['avg_trade_duration_hours']:.1f} hours",
        f"  final equity:     ${m.get('final_equity', 0):,.0f}",
    ]
    return "\n".join(lines)
