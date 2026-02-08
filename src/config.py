"""Configuration management — loads YAML and provides typed access."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "bot_config.yaml"
_config_cache: dict[str, Any] | None = None


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and cache the bot configuration.

    Loads from the given path, or the default configs/bot_config.yaml.
    Environment variable DOMINANCE_BOT_CONFIG can override the path.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if path is None:
        path = os.environ.get("DOMINANCE_BOT_CONFIG", str(_DEFAULT_CONFIG_PATH))

    with open(path, "r") as f:
        _config_cache = yaml.safe_load(f)
    return _config_cache


def get(key: str, default: Any = None) -> Any:
    """Get a dotted config key, e.g. 'safety.shot_quality_threshold'."""
    cfg = load_config()
    parts = key.split(".")
    node = cfg
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def reload_config(path: Path | str | None = None) -> dict[str, Any]:
    """Force reload the config (for tests/hot-reload)."""
    global _config_cache
    _config_cache = None
    return load_config(path)
