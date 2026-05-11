"""Anti-overfitting metrics & cross-validation schemes.

The classical Sharpe ratio is a notoriously optimistic measure when a researcher
has tried many strategies and reports only the best one. This module implements
three tools that correct for that:

1. **Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado (2014).
   Adjusts an observed Sharpe ratio for the number of trials, the non-normality
   of the return distribution (skew & kurtosis), and the sample length. Returns
   the *probability that the true Sharpe is greater than zero*, given everything
   you tried. A high DSR means the strategy probably has a real edge; a low DSR
   means you got lucky on a coin flip.

2. **Probability of Backtest Overfitting (PBO)** via Combinatorially Symmetric
   Cross-Validation — Bailey, Borwein, López de Prado, Zhu (2014).
   Given a matrix of returns from N candidate strategies over T time periods,
   estimates the probability that the *best-performing-in-sample* strategy will
   underperform the *median strategy out-of-sample*. PBO > 50% means selecting
   the best backtest is no better than picking at random. PBO < 10% is healthy.

3. **Combinatorial Purged K-Fold (CPKF)** — López de Prado, "Advances in
   Financial Machine Learning" (2018).
   Replacement for k-fold CV on time series. Splits time into N groups, picks
   k of them as the test fold (C(N,k) combinations), and *purges* training
   samples whose evaluation horizon overlaps the test fold, plus an *embargo*
   on the bars immediately following each test block. Eliminates label leakage
   in walk-forward studies.

References:
    Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for
        Selection Bias, Backtest Overfitting, and Non-Normality" (2014).
    Bailey, Borwein, López de Prado, Zhu, "The Probability of Backtest
        Overfitting" (2014).
    López de Prado, "Advances in Financial Machine Learning" (2018), Ch. 7.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# Euler-Mascheroni constant, used by the expected-max-SR approximation in DSR.
_EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class DeflatedSharpeResult:
    observed_sharpe: float          # SR computed from the chosen strategy (annualized)
    expected_max_sharpe: float      # E[max SR] across n_trials i.i.d. random strategies (annualized)
    deflated_sharpe: float          # P(true SR > 0 | observed) ∈ [0, 1]
    n_trials: int
    n_observations: int
    skew: float
    kurtosis_excess: float

    def summary(self) -> str:
        verdict = ("LIKELY REAL EDGE" if self.deflated_sharpe >= 0.95
                   else "INCONCLUSIVE" if self.deflated_sharpe >= 0.5
                   else "LIKELY OVERFIT")
        return (
            f"Deflated Sharpe Ratio:\n"
            f"  observed SR        {self.observed_sharpe:+.3f}\n"
            f"  expected max SR    {self.expected_max_sharpe:+.3f}  "
            f"(across {self.n_trials} trials, T={self.n_observations})\n"
            f"  return skew        {self.skew:+.3f}\n"
            f"  return excess kurt {self.kurtosis_excess:+.3f}\n"
            f"  DSR = P(SR_true>0) {self.deflated_sharpe:.4f}  → {verdict}"
        )


def _expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """E[max Z_i] for n_trials i.i.d. standard-normal Sharpes, scaled by sqrt(var).

    Closed-form approximation from Bailey & López de Prado (2014):
        E[max SR] ≈ sqrt(var(SR_estimates)) × [(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne))]
    where γ is the Euler-Mascheroni constant.
    """
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    sd = float(np.sqrt(max(sr_variance, 0.0)))
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * np.e))
    return sd * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    returns: Sequence[float] | pd.Series,
    n_trials: int,
    sr_variance: float | None = None,
    periods_per_year: float = 252.0,
    sr_benchmark: float | None = None,
) -> DeflatedSharpeResult:
    """Compute the Deflated Sharpe Ratio for a single strategy's return series.

    Args:
        returns: per-period returns of the chosen strategy (not cumulative).
        n_trials: number of strategy configurations you evaluated before
                  selecting this one (must include every backtest you ran,
                  even informally — that's the whole point).
        sr_variance: variance of the (annualized) Sharpe ratios across the
                  n_trials configurations. If None, defaults to (2/T) which is
                  the asymptotic variance of an i.i.d. Gaussian SR — a
                  conservative lower bound.
        periods_per_year: 252 for daily, 12 for monthly, etc.
        sr_benchmark: SR to test against; defaults to E[max SR] under the null.

    Returns:
        DeflatedSharpeResult with the deflated SR (a probability in [0, 1]).
    """
    r = pd.Series(returns).dropna().astype(float)
    T = len(r)
    if T < 3:
        return DeflatedSharpeResult(0.0, 0.0, 0.0, n_trials, T, 0.0, 0.0)

    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 1e-12:  # treat numerical-noise stddev as degenerate
        return DeflatedSharpeResult(0.0, 0.0, 0.0, n_trials, T, 0.0, 0.0)

    sr_per_period = mu / sd
    sr_annual = sr_per_period * np.sqrt(periods_per_year)

    # Higher moments of the *per-period* returns
    centered = r - mu
    m3 = float((centered ** 3).mean())
    m4 = float((centered ** 4).mean())
    skew = m3 / (sd ** 3) if sd > 0 else 0.0
    kurt_excess = (m4 / (sd ** 4) - 3.0) if sd > 0 else 0.0

    # Variance of Sharpe estimator under non-normality (Mertens, 2002):
    #   Var(SR_hat) ≈ (1 - γ·SR + ((κ-1)/4)·SR²) / (T - 1)
    # We use the *per-period* SR here, since the variance scales with sqrt(T).
    sr_for_var = sr_per_period
    var_sr_estimator = max(
        (1.0 - skew * sr_for_var + (kurt_excess / 4.0) * sr_for_var ** 2) / (T - 1),
        1e-12,
    )

    # Default benchmark = expected max SR under H0
    if sr_variance is None:
        sr_variance = 2.0 / T  # i.i.d. Gaussian lower bound for cross-trial SR variance
    if sr_benchmark is None:
        # Convert benchmark to per-period scale for the standardized stat
        sr_benchmark = _expected_max_sharpe(sr_variance, n_trials) / np.sqrt(periods_per_year)

    # Standardized statistic, then probability
    z = (sr_for_var - sr_benchmark) / np.sqrt(var_sr_estimator)
    dsr = float(norm.cdf(z))

    return DeflatedSharpeResult(
        observed_sharpe=float(sr_annual),
        expected_max_sharpe=float(sr_benchmark * np.sqrt(periods_per_year)),
        deflated_sharpe=dsr,
        n_trials=int(n_trials),
        n_observations=int(T),
        skew=float(skew),
        kurtosis_excess=float(kurt_excess),
    )


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (CSCV)                                  #
# --------------------------------------------------------------------------- #

@dataclass
class PBOResult:
    pbo: float                  # P(best-IS strategy is below-median OOS)
    n_strategies: int
    n_blocks: int
    n_splits: int               # number of (IS, OOS) symmetric splits evaluated
    logits: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        verdict = ("HEALTHY"  if self.pbo < 0.10
                   else "MODERATE OVERFITTING" if self.pbo < 0.30
                   else "SEVERE OVERFITTING")
        return (
            f"Probability of Backtest Overfitting (CSCV):\n"
            f"  strategies tested  {self.n_strategies}\n"
            f"  blocks (S)         {self.n_blocks}\n"
            f"  symmetric splits   {self.n_splits}\n"
            f"  PBO                {self.pbo:.4f}  → {verdict}\n"
            f"  (PBO < 0.1 = good; > 0.5 = best backtest is meaningless)"
        )


def _split_into_blocks(matrix: np.ndarray, n_blocks: int) -> List[np.ndarray]:
    """Split T×N return matrix into n_blocks contiguous T/S × N submatrices."""
    T = matrix.shape[0]
    if T < n_blocks:
        raise ValueError(f"need T >= S blocks; got T={T}, S={n_blocks}")
    # Trim to a multiple of n_blocks so blocks are equal length
    trim = (T // n_blocks) * n_blocks
    return np.array_split(matrix[:trim], n_blocks, axis=0)


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray | pd.DataFrame,
    n_blocks: int = 16,
    score_fn=None,
) -> PBOResult:
    """Estimate PBO via Combinatorially Symmetric Cross-Validation.

    Args:
        returns_matrix: T×N matrix where each column is one strategy's per-period
                       returns. T = number of periods, N = number of strategies.
        n_blocks: S, must be even. Default 16 follows Bailey et al. Smaller →
                 fewer splits but more data per fold; larger → more splits but
                 noisier per-fold scores. Each split puts S/2 blocks in IS,
                 the other S/2 in OOS.
        score_fn: function(returns_2d) → array of per-column scores.
                 Default is Sharpe ratio (mean/std per column).

    Returns:
        PBOResult with PBO ∈ [0, 1] and the logit distribution.
    """
    if isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = returns_matrix.values
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("returns_matrix must be 2D (T periods × N strategies)")
    T, N = M.shape
    if N < 2:
        raise ValueError(f"need >=2 strategies to estimate PBO; got N={N}")
    if n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even; got {n_blocks}")

    if score_fn is None:
        def score_fn(arr: np.ndarray) -> np.ndarray:
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=0)
            sd_safe = np.where(sd > 0, sd, np.nan)
            sr = mu / sd_safe
            return np.nan_to_num(sr, nan=-np.inf)

    blocks = _split_into_blocks(M, n_blocks)
    S = n_blocks
    half = S // 2
    block_indices = list(range(S))

    logits: list[float] = []
    n_below_median = 0
    n_total = 0

    for is_blocks in combinations(block_indices, half):
        oos_blocks = [b for b in block_indices if b not in is_blocks]
        is_data = np.concatenate([blocks[b] for b in is_blocks], axis=0)
        oos_data = np.concatenate([blocks[b] for b in oos_blocks], axis=0)

        is_scores = score_fn(is_data)
        oos_scores = score_fn(oos_data)

        # Strategy n* selected on IS
        n_star = int(np.argmax(is_scores))

        # OOS rank of n_star, expressed as percentile in [1/(N+1), N/(N+1)]
        # to avoid logit blowup at the extremes.
        ranks = np.argsort(np.argsort(oos_scores))  # 0..N-1
        rank_star = ranks[n_star]
        w = (rank_star + 1) / (N + 1)  # smoothed percentile in (0, 1)

        logit = float(np.log(w / (1.0 - w)))
        logits.append(logit)
        if w < 0.5:
            n_below_median += 1
        n_total += 1

    pbo = n_below_median / n_total if n_total else 0.0
    return PBOResult(
        pbo=float(pbo),
        n_strategies=int(N),
        n_blocks=int(n_blocks),
        n_splits=int(n_total),
        logits=np.asarray(logits),
    )


# --------------------------------------------------------------------------- #
# Combinatorial Purged K-Fold                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class CPKFSplit:
    """One (train, test) index split with purging applied."""
    train: np.ndarray  # indices into the original observation array
    test: np.ndarray
    test_groups: Tuple[int, ...]  # which group ids form the test fold


def combinatorial_purged_splits(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    eval_horizon: int = 1,
) -> List[CPKFSplit]:
    """Generate combinatorial purged k-fold (train, test) splits.

    Args:
        n_samples: total number of time-ordered observations.
        n_groups: number of contiguous time-blocks (N). C(N,k) splits returned.
        n_test_groups: how many blocks per test fold (k). e.g. N=6, k=2 → 15 splits.
        embargo_pct: fraction of n_samples to drop from train *after* each test
                    block, to prevent leakage through autocorrelation. 1% is typical.
        eval_horizon: how many bars after sample t its label/return depends on.
                    Train samples whose [t, t+horizon] overlaps any test block
                    are purged. Use 1 for trade-by-trade backtests, larger for
                    forecasts with longer evaluation horizons.

    Returns:
        List of CPKFSplit; one entry per combination of test blocks.
    """
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("need n_groups >= 2 and 0 < n_test_groups < n_groups")
    if n_samples < n_groups:
        raise ValueError(f"need n_samples >= n_groups; got {n_samples} < {n_groups}")

    # Assign each sample to a group based on its time index.
    group_size = n_samples / n_groups
    group_of = np.minimum((np.arange(n_samples) / group_size).astype(int), n_groups - 1)
    embargo = max(0, int(round(n_samples * embargo_pct)))

    splits: list[CPKFSplit] = []
    for test_groups in combinations(range(n_groups), n_test_groups):
        test_mask = np.isin(group_of, test_groups)
        test_idx = np.where(test_mask)[0]

        # Purge: drop any training sample whose [t, t+eval_horizon] intersects test.
        train_mask = ~test_mask
        # Vectorized purge: for each training index i, check if any of
        # [i, i+eval_horizon-1] falls within test_mask.
        if eval_horizon > 1:
            # Forward-fill: a training point is purged if test_mask[i..i+h-1] any True.
            # Equivalent to: rolling-window OR over the next h positions.
            rolled = np.zeros(n_samples, dtype=bool)
            for off in range(eval_horizon):
                rolled[: n_samples - off] |= test_mask[off:]
            train_mask &= ~rolled
        else:
            # horizon=1: only the test bars themselves are dropped (already excluded)
            pass

        # Embargo: drop a train window of length `embargo` after each contiguous
        # test block, since their labels still leak forward.
        if embargo > 0 and test_idx.size:
            # Find the end of each contiguous test run, then embargo the next bars.
            gaps = np.where(np.diff(test_idx) > 1)[0]
            run_ends = np.append(test_idx[gaps], test_idx[-1])
            for end in run_ends:
                emb_start = end + 1
                emb_end = min(n_samples, emb_start + embargo)
                if emb_start < emb_end:
                    train_mask[emb_start:emb_end] = False

        train_idx = np.where(train_mask)[0]
        splits.append(CPKFSplit(
            train=train_idx,
            test=test_idx,
            test_groups=tuple(test_groups),
        ))
    return splits


# --------------------------------------------------------------------------- #
# Convenience: build a returns matrix from a sensitivity scan                 #
# --------------------------------------------------------------------------- #

def returns_matrix_from_sensitivity(
    rows: Iterable[dict],
    equity_curves: Iterable[pd.Series],
    resample: str = "1D",
) -> pd.DataFrame:
    """Stack per-config equity curves into a T×N daily-return matrix.

    Each sensitivity row's equity curve is resampled to `resample`, converted
    to pct-change, then aligned on a common date index. Missing days are filled
    with 0 (flat day).
    """
    series: list[pd.Series] = []
    for i, eq in enumerate(equity_curves):
        if eq is None or len(eq) < 2:
            continue
        eq_r = eq.resample(resample).last().ffill()
        rets = eq_r.pct_change().dropna()
        rets.name = f"cfg_{i}"
        series.append(rets)
    if not series:
        return pd.DataFrame()
    df = pd.concat(series, axis=1).fillna(0.0)
    return df
