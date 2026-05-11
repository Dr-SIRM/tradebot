"""Centralized logging configuration."""
from __future__ import annotations
import logging
import logging.handlers
import os
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", file: str = "logs/tradebot.log",
                  rotate_mb: int = 50) -> logging.Logger:
    """Configure root logger with console + rotating file handlers."""
    Path(file).parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s.%(msecs)03d %(levelname)-8s %(name)-22s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear any existing handlers (idempotent setup)
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Rotating file
    fh = logging.handlers.RotatingFileHandler(
        file, maxBytes=rotate_mb * 1024 * 1024, backupCount=5
    )
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Tame noisy libraries
    for noisy in ["urllib3", "asyncio", "websockets.client", "ccxt"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
