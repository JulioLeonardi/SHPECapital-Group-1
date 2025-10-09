from __future__ import annotations
from pathlib import Path
import yaml
from typing import Any, Dict


CFG_CACHE: Dict[str, Dict[str, Any]] = {}


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config into a dict (cached by absolute path)."""
    p = Path(path).resolve()
    if str(p) in CFG_CACHE:
        return CFG_CACHE[str(p)]
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    CFG_CACHE[str(p)] = cfg
    return cfg