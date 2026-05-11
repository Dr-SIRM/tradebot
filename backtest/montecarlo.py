"""Monte Carlo simulation: bootstrap resample trades to assess strategy robustness.

We resample with replacement from the realized trade R-multiples to build
synthetic equity curves. This stress-tests sequence-of-returns risk and
gives percentile bands on final equity, max drawdown, and Sharpe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from utils.types import Trade


@dataclass
class MonteCarloResult:
    n_runs: int
    starting_equity: float
    final_equity_mean: float
    final_equity_median: float
    final_equity_p05: float
    final_equity_p95: float
    max_dd_mean: float
    max_dd_median: float
    max_dd_p95: float  # 95th percentile of max DD = "bad case" drawdown
    prob_profitable: float
    prob_ruin: float  # P(equity touches 50% of start)
    final_equities: np.ndarray = field(default_factory=lambda: np.array([]))
    max_drawdowns: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        return (
            f"Monte Carlo ({self.n_runs} runs):\n"
            f"  Final Equity   mean={self.final_equity_mean:,.0f}  "
            f"median={self.final_equity_median:,.0f}  "
            f"p05={self.final_equity_p05:,.0f}  p95={self.final_equity_p95:,.0f}\n"
            f"  Max Drawdown   mean={self.max_dd_mean*100:.2f}%  "
            f"p95={self.max_dd_p95*100:.2f}% (bad case)\n"
            f"  P(profitable)  {self.prob_profitable*100:.1f}%\n"
            f"  P(ruin@50%)    {self.prob_ruin*100:.2f}%"
        )


def _trade_pct_returns(trades: Sequence[Trade], starting_equity: float) -> np.ndarray:
    """Convert trades to percentage returns on equity at time of trade.

    We use the realized $ pnl divided by the starting equity as a stable
    approximation. For more accurate path dependency you'd compound through
    the actual trade sequence, but for bootstrap MC the assumption of
    return-distribution stationarity is the standard approach.
    """
    if not trades:
        return np.array([])
    return np.array([t.pnl / starting_equity for t in trades])


def run_monte_carlo(
    trades: Sequence[Trade],
    starting_equity: float = 100_000.0,
    n_runs: int = 1000,
    n_trades_per_run: int | None = None,
    ruin_threshold: float = 0.5,
    seed: int | None = None,
) -> MonteCarloResult:
    """Bootstrap-resample trade returns to build synthetic equity curves.

    Args:
        trades: realized trades from a backtest.
        starting_equity: starting capital for each simulated path.
        n_runs: number of Monte Carlo paths.
        n_trades_per_run: trades per path (default: same as input).
        ruin_threshold: equity fraction below which we count "ruin"
                        (e.g. 0.5 = touched 50% drawdown).
        seed: RNG seed for reproducibility.
    """
    if not trades:
        return MonteCarloResult(
            n_runs=0,
            starting_equity=starting_equity,
            final_equity_mean=starting_equity,
            final_equity_median=starting_equity,
            final_equity_p05=starting_equity,
            final_equity_p95=starting_equity,
            max_dd_mean=0.0,
            max_dd_median=0.0,
            max_dd_p95=0.0,
            prob_profitable=0.0,
            prob_ruin=0.0,
        )

    rng = np.random.default_rng(seed)
    pct_returns = _trade_pct_returns(trades, starting_equity)
    n_trades = n_trades_per_run if n_trades_per_run is not None else len(trades)

    final_equities = np.empty(n_runs)
    max_drawdowns = np.empty(n_runs)
    n_ruined = 0
    ruin_level = starting_equity * ruin_threshold

    for i in range(n_runs):
        sample = rng.choice(pct_returns, size=n_trades, replace=True)
        # Compound multiplicatively on starting equity. Each trade adds
        # pct_return * starting_equity in dollar terms (additive bootstrap),
        # which is the standard formulation for fixed-fractional MC.
        equity_path = starting_equity + np.cumsum(sample * starting_equity)
        equity_path = np.concatenate(([starting_equity], equity_path))

        final_equities[i] = equity_path[-1]
        running_peak = np.maximum.accumulate(equity_path)
        dd = (running_peak - equity_path) / running_peak
        max_drawdowns[i] = float(dd.max())

        if equity_path.min() <= ruin_level:
            n_ruined += 1

    return MonteCarloResult(
        n_runs=n_runs,
        starting_equity=starting_equity,
        final_equity_mean=float(np.mean(final_equities)),
        final_equity_median=float(np.median(final_equities)),
        final_equity_p05=float(np.percentile(final_equities, 5)),
        final_equity_p95=float(np.percentile(final_equities, 95)),
        max_dd_mean=float(np.mean(max_drawdowns)),
        max_dd_median=float(np.median(max_drawdowns)),
        max_dd_p95=float(np.percentile(max_drawdowns, 95)),
        prob_profitable=float(np.mean(final_equities > starting_equity)),
        prob_ruin=n_ruined / n_runs,
        final_equities=final_equities,
        max_drawdowns=max_drawdowns,
    )
