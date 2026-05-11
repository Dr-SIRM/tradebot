"""Entry point for live (or paper) trading.

Defaults to PAPER mode. Set TRADEBOT_LIVE=1 in the environment to enable
live trading. Live mode also requires the chosen broker(s) to have valid
credentials in the YAML config (or env vars referenced by the config).

Usage:
    python run_live.py                        # paper mode, default config
    python run_live.py --config config/my.yaml
    TRADEBOT_LIVE=1 python run_live.py        # LIVE — real money risk
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from typing import Any, Dict

from alerts.notifier import Notifier
from execution.broker_base import BrokerBase
from execution.paper import PaperBroker
from orchestrator import Orchestrator
from utils.config import load_config
from utils.logging import setup_logging, get_logger

log = get_logger(__name__)


def _build_brokers(cfg: Dict[str, Any], live: bool) -> Dict[str, BrokerBase]:
    """Construct broker instances. Always returns a paper broker; if live
    mode is requested, also constructs the real brokers configured in
    cfg['brokers']."""
    brokers: Dict[str, BrokerBase] = {}

    starting_equity = float(cfg["account"]["starting_equity"])

    # Always have a paper broker available
    brokers["paper"] = PaperBroker(cfg, starting_equity)

    # In paper mode, route every symbol through the paper broker regardless of
    # what its symbol cfg requested.
    if not live:
        for sym_cfg in cfg["universe"]:
            sym_cfg["broker"] = "paper"
        return brokers

    # LIVE — build the real brokers referenced in the universe
    needed_brokers = {sym["broker"] for sym in cfg["universe"]} - {"paper"}

    for name in needed_brokers:
        broker_cfg = cfg["brokers"].get(name)
        if broker_cfg is None:
            log.error("LIVE mode: no config block for broker '%s'", name)
            continue
        try:
            if name == "alpaca":
                from execution.alpaca import AlpacaBroker
                brokers[name] = AlpacaBroker(broker_cfg)
            elif name in ("binance", "bybit", "coinbase"):
                from execution.binance import BinanceBroker
                brokers[name] = BinanceBroker(broker_cfg)
            elif name == "oanda":
                from execution.oanda import OandaBroker
                brokers[name] = OandaBroker(broker_cfg)
            else:
                log.error("LIVE mode: unknown broker '%s'", name)
        except Exception as e:
            log.exception("LIVE mode: failed to build broker %s: %s", name, e)

    return brokers


async def _main_async(cfg_path: str) -> int:
    cfg = load_config(cfg_path)
    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        file=log_cfg.get("file", "logs/tradebot.log"),
        rotate_mb=log_cfg.get("rotate_mb", 50),
    )

    live_env = os.environ.get("TRADEBOT_LIVE", "0") == "1"
    cfg_mode = cfg.get("mode", "paper").lower()
    live = live_env and cfg_mode == "live"

    if live_env and cfg_mode != "live":
        log.warning("TRADEBOT_LIVE=1 set but config mode=%s; staying in paper mode", cfg_mode)
    if live:
        log.warning("=" * 60)
        log.warning("LIVE TRADING ENABLED — REAL MONEY AT RISK")
        log.warning("=" * 60)
        # Final confirmation in interactive sessions
        if sys.stdin.isatty():
            ans = input("Type 'YES' to confirm live trading: ").strip()
            if ans != "YES":
                log.warning("aborted by user")
                return 1

    brokers = _build_brokers(cfg, live)
    if not brokers:
        log.error("no brokers built — aborting")
        return 2

    # Connect brokers
    for name, b in brokers.items():
        try:
            await b.connect()
            log.info("connected broker: %s", name)
        except Exception as e:
            log.exception("broker %s connect failed: %s", name, e)
            return 3

    orch = Orchestrator(cfg)
    orch.attach_brokers(brokers)

    # Signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_signals = (signal.SIGINT, signal.SIGTERM)
    for s in stop_signals:
        try:
            loop.add_signal_handler(s, lambda: asyncio.create_task(orch.stop()))
        except NotImplementedError:
            # Windows — signals not fully supported
            pass

    try:
        await orch.run()
    except KeyboardInterrupt:
        await orch.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the trading bot.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()
    try:
        return asyncio.run(_main_async(args.config))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
