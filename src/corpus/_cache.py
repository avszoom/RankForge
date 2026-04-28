"""Tiny disk-cache helper shared by the three generators."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2] / "data" / "_raw"


def cache_path(phase: str, key: str) -> Path:
    return ROOT / phase / f"{key}.json"


def load_cached(phase: str, key: str) -> dict[str, Any] | None:
    p = cache_path(phase, key)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def save_cache(phase: str, key: str, data: dict[str, Any]) -> None:
    p = cache_path(phase, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
