"""Logical dataset names used by the analysis scripts (review-mirror version)."""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def resolve(name: str) -> Path:
    entry = json.loads((_ROOT / "datasets/registry.json").read_text())["datasets"][name]
    path = _ROOT / "datasets" / entry["path"]
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path} is not included in this release ({entry.get('note', '')})")
    return path
