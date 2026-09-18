"""Config loading + path resolution.

Follows the Swiggy DS repo pattern: a single YAML config drives every
pipeline stage; repo-relative paths so the project runs from any cwd.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.constants import REPO_ROOT


def _resolve_paths(node: Any) -> Any:
    """Recursively resolve `data/...`-style relative paths against REPO_ROOT."""
    if isinstance(node, dict):
        return {k: _resolve_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_paths(v) for v in node]
    if isinstance(node, str):
        if node.startswith(("http://", "https://")):
            return node  # urls are not filesystem paths — never resolve them
        candidate = Path(node)
        if not candidate.is_absolute() and "/" in node:
            return (REPO_ROOT / node).as_posix()
    return node


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/config.yaml and return a plain nested dict.

    Relative data/model/report paths are expanded to absolute repo-root
    paths so stages never depend on the caller's working directory.
    """
    cfg_path = Path(path) if path else REPO_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resolved = _resolve_paths(cfg)

    # Deep-copy guard: callers mutate sections (e.g. candidate params) freely.
    return copy.deepcopy(resolved)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Dot-access helper: get(cfg, 'training.quantiles') -> [0.1, 0.5, 0.9]."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
