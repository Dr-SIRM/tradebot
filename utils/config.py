"""YAML config loader with ${ENV_VAR} substitution."""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any
import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _substitute_env(obj: Any) -> Any:
    if isinstance(obj, str):
        def repl(m):
            var = m.group(1)
            return os.environ.get(var, "")
        return _ENV_PATTERN.sub(repl, obj)
    if isinstance(obj, dict):
        return {k: _substitute_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env(v) for v in obj]
    return obj


def load_config(path: str | Path) -> dict:
    """Load YAML and resolve ${ENV} references."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return _substitute_env(raw)
