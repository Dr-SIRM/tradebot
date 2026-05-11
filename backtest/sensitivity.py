"""Parameter sensitivity analysis: grid-scan strategy params to detect overfitting.

A strategy whose performance collapses outside a narrow parameter pocket is
likely overfit. We sweep a grid and report the distribution of metrics.
Stable strategies show similar Sharpe/profit-factor across neighboring params.
"""
from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from backtest.engine import Backtester
from backtest.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    grid_keys: List[str]
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary(self) -> str:
        if not self.rows:
            return "Sensitivity: no results."
        df = self.to_dataframe()
        if "sharpe" not in df.columns:
            return f"Sensitivity ({len(self.rows)} configs): no usable runs."
        sharpes = df["sharpe"].dropna().values
        pfs = df["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(sharpes) == 0:
            return f"Sensitivity ({len(self.rows)} configs): no completed runs."
        return (
            f"Sensitivity ({len(self.rows)} configs over {self.grid_keys}):\n"
            f"  Sharpe        mean={np.mean(sharpes):.2f}  std={np.std(sharpes):.2f}  "
            f"min={np.min(sharpes):.2f}  max={np.max(sharpes):.2f}\n"
            f"  Profit Factor mean={np.mean(pfs):.2f}  std={np.std(pfs):.2f}\n"
            f"  Stability (mean/std Sharpe): "
            f"{np.mean(sharpes) / (np.std(sharpes) + 1e-9):.2f}"
        )


def _expand_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _set_param(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set cfg[a][b][c] = value from dotted_key 'a.b.c'.

    Bare keys (no dot) are assumed to live under strategy.momentum since
    that's the most common sensitivity target.
    """
    if "." not in dotted_key:
        dotted_key = f"strategy.momentum.{dotted_key}"
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node:
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def run_sensitivity(
    ltf: pd.DataFrame,
    htf: pd.DataFrame,
    base_cfg: Dict[str, Any],
    grid: Dict[str, Sequence[Any]],
    symbol: str,
    asset_class: str,
    timeframe_low: str = "5m",
) -> SensitivityResult:
    """Grid-scan strategy parameters and report metric stability.

    Args:
        ltf: lower-timeframe bars.
        htf: higher-timeframe bars.
        base_cfg: full config dict (deep-copied per run).
        grid: dotted-key -> list of values, e.g.
              {"strategy.momentum.ema_fast": [8, 9, 10, 11, 12]}.
    """
    combos = _expand_grid(grid)
    result = SensitivityResult(grid_keys=list(grid.keys()))
    starting_equity = base_cfg["backtest"]["initial_equity"]

    logger.info("Sensitivity scan: %d configs over %s", len(combos), list(grid.keys()))

    for i, combo in enumerate(combos):
        cfg = copy.deepcopy(base_cfg)
        for k, v in combo.items():
            _set_param(cfg, k, v)

        try:
            bt = Backtester(cfg)
            trades, _equity = bt.run(ltf, htf, symbol, asset_class, timeframe_low)
            metrics = compute_metrics(trades, starting_equity)
            row: Dict[str, Any] = {**combo}
            row.update({
                "n_trades": metrics["trades"],
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "profit_factor": metrics["profit_factor"],
                "max_dd_pct": metrics["max_drawdown_pct"],
                "win_rate": metrics["win_rate"],
                "expectancy_R": metrics["expectancy_R"],
                "total_return_pct": metrics["total_return_pct"],
                "final_equity": metrics.get("final_equity", starting_equity),
            })
            result.rows.append(row)
        except Exception as e:
            logger.exception("Sensitivity run %d/%d failed: %s", i + 1, len(combos), e)
            result.rows.append({**combo, "error": str(e)})

        if (i + 1) % 5 == 0 or (i + 1) == len(combos):
            logger.info("  progress %d/%d", i + 1, len(combos))

    return result
