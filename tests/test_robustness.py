"""Tests for backtest.robustness — Deflated Sharpe, PBO, CPKF."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.robustness import (
    CPKFSplit,
    combinatorial_purged_splits,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    returns_matrix_from_sensitivity,
)


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio                                                       #
# --------------------------------------------------------------------------- #

def test_dsr_strong_signal_is_real():
    """A genuinely strong return series with few trials should yield high DSR."""
    rng = np.random.default_rng(0)
    # mu/sd ~= 0.15 daily SR → annualized SR ≈ 2.4 — clearly real
    rets = rng.normal(loc=0.0015, scale=0.01, size=750)
    dsr = deflated_sharpe_ratio(rets, n_trials=1, periods_per_year=252)
    assert dsr.observed_sharpe > 1.5
    assert dsr.deflated_sharpe > 0.95


def test_dsr_zero_signal_is_not_certified():
    """Pure noise with 1 trial should never reach the 'real edge' threshold."""
    rng = np.random.default_rng(1)
    # Average across many noise samples: DSR should center on ~0.5
    dsrs = []
    for seed in range(30):
        r = np.random.default_rng(seed).normal(0.0, 0.01, size=500)
        dsrs.append(deflated_sharpe_ratio(r, n_trials=1).deflated_sharpe)
    mean_dsr = float(np.mean(dsrs))
    assert 0.4 < mean_dsr < 0.6  # centered around 0.5 across seeds
    # And no individual sample should hit the "real edge" bar
    assert max(dsrs) < 0.95


def test_dsr_punishes_many_trials():
    """The more configs tried, the lower the deflated SR for the same observed SR."""
    rng = np.random.default_rng(2)
    rets = rng.normal(loc=0.001, scale=0.01, size=500)
    dsr_1 = deflated_sharpe_ratio(rets, n_trials=1)
    dsr_1000 = deflated_sharpe_ratio(rets, n_trials=1000)
    # Same observed SR, but the benchmark grows with n_trials
    assert dsr_1.observed_sharpe == pytest.approx(dsr_1000.observed_sharpe)
    assert dsr_1000.expected_max_sharpe > dsr_1.expected_max_sharpe
    assert dsr_1000.deflated_sharpe < dsr_1.deflated_sharpe


def test_dsr_handles_degenerate_input():
    """Empty / constant / tiny inputs should not raise."""
    dsr = deflated_sharpe_ratio([], n_trials=1)
    assert dsr.deflated_sharpe == 0.0
    dsr = deflated_sharpe_ratio([0.01] * 100, n_trials=1)  # zero std
    assert dsr.deflated_sharpe == 0.0


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting                                         #
# --------------------------------------------------------------------------- #

def test_pbo_pure_noise_is_high():
    """When all strategies are i.i.d. noise, PBO should be ~50% (random)."""
    rng = np.random.default_rng(3)
    T, N = 800, 30
    M = rng.normal(0.0, 0.01, size=(T, N))
    pbo = probability_of_backtest_overfitting(M, n_blocks=10)
    # With pure noise, the best in-sample strategy is no better than random OOS,
    # so PBO should be roughly 50% (allow wide band: 0.3..0.7).
    assert 0.3 < pbo.pbo < 0.7
    assert pbo.n_strategies == N
    assert pbo.n_splits == 252  # C(10, 5)


def test_pbo_real_signal_is_low():
    """If one strategy has a true edge, the best-IS is almost always best-OOS."""
    rng = np.random.default_rng(4)
    T, N = 600, 20
    M = rng.normal(0.0, 0.01, size=(T, N))
    # Inject a strong edge into strategy 0
    M[:, 0] = rng.normal(0.005, 0.01, size=T)
    pbo = probability_of_backtest_overfitting(M, n_blocks=10)
    # Best IS should consistently be best OOS, so PBO < 0.2
    assert pbo.pbo < 0.2


def test_pbo_rejects_odd_block_count():
    rng = np.random.default_rng(5)
    M = rng.normal(0, 1, size=(200, 5))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(M, n_blocks=7)


def test_pbo_rejects_single_strategy():
    rng = np.random.default_rng(6)
    M = rng.normal(0, 1, size=(200, 1))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(M, n_blocks=4)


# --------------------------------------------------------------------------- #
# Combinatorial Purged K-Fold                                                 #
# --------------------------------------------------------------------------- #

def test_cpkf_split_counts_and_disjoint():
    """C(6, 2) = 15 splits; train and test never overlap."""
    splits = combinatorial_purged_splits(
        n_samples=600, n_groups=6, n_test_groups=2,
        embargo_pct=0.01, eval_horizon=1,
    )
    assert len(splits) == 15
    for s in splits:
        assert isinstance(s, CPKFSplit)
        assert len(np.intersect1d(s.train, s.test)) == 0


def test_cpkf_embargo_removes_post_test_bars():
    """The first bars after each test block should be missing from train."""
    splits = combinatorial_purged_splits(
        n_samples=1000, n_groups=5, n_test_groups=1,
        embargo_pct=0.02, eval_horizon=1,
    )
    embargo_n = int(round(1000 * 0.02))
    for s in splits:
        # Find the end of the test block; check the next `embargo_n` are not in train
        test_end = s.test.max()
        post = np.arange(test_end + 1, min(1000, test_end + 1 + embargo_n))
        if post.size:
            assert len(np.intersect1d(s.train, post)) == 0


def test_cpkf_horizon_purges_overlapping_labels():
    """With eval_horizon > 1, training bars whose horizon overlaps test get purged."""
    splits = combinatorial_purged_splits(
        n_samples=500, n_groups=5, n_test_groups=1,
        embargo_pct=0.0, eval_horizon=10,
    )
    for s in splits:
        test_start = s.test.min()
        # Any training bar at [test_start-10+1, test_start-1] would have its
        # 10-bar horizon overlap the first test bar; those should be purged.
        risky = np.arange(max(0, test_start - 9), test_start)
        assert len(np.intersect1d(s.train, risky)) == 0


def test_cpkf_validates_args():
    with pytest.raises(ValueError):
        combinatorial_purged_splits(n_samples=100, n_groups=1, n_test_groups=1)
    with pytest.raises(ValueError):
        combinatorial_purged_splits(n_samples=100, n_groups=5, n_test_groups=5)
    with pytest.raises(ValueError):
        combinatorial_purged_splits(n_samples=3, n_groups=5, n_test_groups=1)


# --------------------------------------------------------------------------- #
# returns_matrix_from_sensitivity                                             #
# --------------------------------------------------------------------------- #

def test_returns_matrix_aligns_curves():
    idx = pd.date_range("2024-01-01", periods=30, freq="1D", tz="UTC")
    eq1 = pd.Series(np.linspace(100_000, 110_000, 30), index=idx)
    eq2 = pd.Series(np.linspace(100_000, 95_000, 30), index=idx[5:].union(idx[:5]))
    mat = returns_matrix_from_sensitivity(
        rows=[{}, {}], equity_curves=[eq1, eq2], resample="1D"
    )
    assert mat.shape[1] == 2
    assert mat.shape[0] >= 25
    # Equity 1 increased, equity 2 decreased on average
    assert mat.iloc[:, 0].sum() > 0
    assert mat.iloc[:, 1].sum() < 0


def test_returns_matrix_handles_empty_inputs():
    mat = returns_matrix_from_sensitivity(rows=[], equity_curves=[])
    assert mat.empty
