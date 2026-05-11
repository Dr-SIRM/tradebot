"""XGBoost model wrapper. Lightweight load/save/predict — training lives in train.py."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

from utils.logging import get_logger
from utils.types import Signal
from ml.features import build_features, FEATURE_NAMES

log = get_logger(__name__)


class TradeScorer:
    def __init__(self, model_path: str | Path | None = None,
                 feature_lookback: int = 50):
        self.model = None
        self.feature_lookback = feature_lookback
        if model_path:
            self.load(model_path)

    def load(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            log.warning("ML model not found at %s — scoring disabled", p)
            return False
        try:
            with p.open("rb") as f:
                self.model = pickle.load(f)
            log.info("loaded ML model from %s", p)
            return True
        except Exception as e:
            log.error("failed to load ML model: %s", e)
            return False

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self.model, f)
        log.info("saved ML model to %s", p)

    def score(self, signal: Signal, ltf, htf) -> float | None:
        """Return probability the trade is profitable, in [0, 1]. None if model
        not loaded or features unavailable."""
        if self.model is None:
            return None
        feats = build_features(signal, ltf, htf, self.feature_lookback)
        if feats is None:
            return None
        try:
            proba = self.model.predict_proba(feats.reshape(1, -1))[0, 1]
            return float(proba)
        except Exception as e:
            log.warning("ML scoring failed: %s", e)
            return None
