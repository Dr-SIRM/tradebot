"""Train the trade scorer.

Pipeline:
  1. Run a backtest with NO ML scoring across one or more symbols.
  2. Capture every signal generated, plus the trade outcome (win=1, loss=0).
  3. Build features at signal time.
  4. Train XGBoost classifier with cross-validated grid search.
  5. Save to ml/model.pkl.

Usage:
    python -m ml.train --config config/default.yaml --data data/btc_5m.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure the project root is importable when run via `python ml/train.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_config
from utils.logging import setup_logging, get_logger
from data.feed import load_historical
from data.indicators import add_all_indicators
from strategy.selector import StrategySelector
from ml.features import build_features

log = get_logger(__name__)


def collect_training_data(cfg: dict, ltf_df: pd.DataFrame, htf_df: pd.DataFrame,
                            symbol: str, asset_class: str, timeframe_low: str,
                            future_bars: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Walk the data, generate signals, label by future return.

    Label = 1 if the trade hit its TP before its stop within `future_bars`,
            0 if it hit stop first,
            -1 (skip) if neither (we don't train on indeterminate outcomes).
    """
    ltf_ind = add_all_indicators(ltf_df, cfg)
    htf_ind = add_all_indicators(htf_df, cfg)
    selector = StrategySelector(cfg)

    feats_list: list[np.ndarray] = []
    labels: list[int] = []

    warmup = cfg["data"]["warmup_bars"]
    for i in range(warmup, len(ltf_ind) - future_bars):
        ltf_slice = ltf_ind.iloc[: i + 1]
        # Align HTF — use only HTF bars up to this LTF timestamp
        cur_ts = ltf_slice.index[-1]
        htf_slice = htf_ind.loc[:cur_ts]
        if len(htf_slice) < 60:
            continue

        regime, signal = selector.select(htf_slice, ltf_slice, symbol, asset_class, timeframe_low)
        if signal is None:
            continue

        feat_vec = build_features(signal, ltf_slice, htf_slice, cfg["ml"]["feature_lookback"])
        if feat_vec is None:
            continue

        # Walk forward bar-by-bar to determine outcome
        outcome = -1
        for j in range(i + 1, min(i + 1 + future_bars, len(ltf_ind))):
            bar = ltf_ind.iloc[j]
            if signal.side.value == "long":
                if bar["low"] <= signal.stop:
                    outcome = 0
                    break
                if bar["high"] >= signal.take_profit:
                    outcome = 1
                    break
            else:
                if bar["high"] >= signal.stop:
                    outcome = 0
                    break
                if bar["low"] <= signal.take_profit:
                    outcome = 1
                    break

        if outcome == -1:
            continue

        feats_list.append(feat_vec)
        labels.append(outcome)

    if not feats_list:
        return np.empty((0, 0)), np.empty((0,))
    return np.vstack(feats_list), np.asarray(labels)


def train_xgboost(X: np.ndarray, y: np.ndarray, model_out: Path) -> None:
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score

    if len(X) < 100:
        raise ValueError(f"need >= 100 samples to train, got {len(X)}")

    log.info("training XGBoost on %d samples (positives=%d, negatives=%d)",
             len(X), int(y.sum()), int(len(y) - y.sum()))

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    log.info("CV AUC: %.3f ± %.3f (folds: %s)",
             auc_scores.mean(), auc_scores.std(), [f"{s:.3f}" for s in auc_scores])

    if auc_scores.mean() < 0.55:
        log.warning("CV AUC < 0.55 — model is barely better than random. "
                    "Use the scorer with skepticism.")

    model.fit(X, y)
    train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    log.info("train AUC: %.3f", train_auc)

    import pickle
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(model, f)
    log.info("saved model to %s", model_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--data-ltf", required=True, help="LTF OHLCV CSV/parquet")
    p.add_argument("--data-htf", required=True, help="HTF OHLCV CSV/parquet")
    p.add_argument("--symbol", required=True)
    p.add_argument("--asset-class", default="crypto")
    p.add_argument("--timeframe-low", default="5m")
    p.add_argument("--future-bars", type=int, default=20)
    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["logging"]["level"], cfg["logging"]["file"])

    ltf = load_historical(args.data_ltf)
    htf = load_historical(args.data_htf)

    X, y = collect_training_data(cfg, ltf, htf, args.symbol, args.asset_class,
                                  args.timeframe_low, args.future_bars)
    log.info("collected %d labeled signals", len(X))

    if len(X) == 0:
        log.error("no labeled signals — try a longer dataset or relax strategy params")
        return

    out = Path(cfg["ml"]["model_path"])
    train_xgboost(X, y, out)


if __name__ == "__main__":
    main()
