"""Correlation/exposure manager.

Two checks:
  1. Cluster check: if a symbol's cluster already has an open position,
     reject the new entry (counts as one exposure).
  2. Asset-class cap: each asset class (crypto/equity/forex/commodity) has
     a max simultaneous open position count.
"""
from __future__ import annotations
from utils.types import Position


class CorrelationManager:
    def __init__(self, corr_cfg: dict):
        self.clusters: list[set[str]] = [set(c) for c in corr_cfg.get("clusters", [])]
        self.class_caps: dict[str, int] = corr_cfg.get("asset_class_caps", {})

    def cluster_for(self, symbol: str) -> set[str] | None:
        for c in self.clusters:
            if symbol in c:
                return c
        return None

    def can_open(self, symbol: str, asset_class: str,
                 open_positions: list[Position]) -> tuple[bool, str]:
        # Cluster check
        cluster = self.cluster_for(symbol)
        if cluster:
            for pos in open_positions:
                if pos.symbol in cluster and pos.symbol != symbol:
                    return False, f"correlated with open {pos.symbol} (same cluster)"

        # Asset class cap
        cap = self.class_caps.get(asset_class)
        if cap is not None:
            count = sum(1 for p in open_positions if p.asset_class == asset_class)
            if count >= cap:
                return False, f"asset class cap reached: {asset_class}={count}/{cap}"
        return True, ""
