# PWW-DISCLOSURE: public
"""Task-level context for the modifiers a single frame cannot show.

Load and how often it is handled are properties of the task, not of one image.
They come from `standards/task_context.json`, keyed by the dataset's process
label, and every caller passes them to `score_all` / `mc_score_all` so the
deterministic scores and the Monte Carlo samples use the same context.
"""
import json
from functools import lru_cache
from pathlib import Path

TABLE = Path(__file__).resolve().parents[1] / "standards" / "task_context.json"


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8"))


def task_context(process_label: str | None) -> dict:
    """Keyword arguments for score_all / mc_score_all for one task."""
    t = _table()
    task = t["tasks"].get(process_label or "", {})
    return {"static_minute": bool(t["static_or_repeated"]),
            "load_kg": float(task.get("load_kg", 0.0)),
            "load_repeated": bool(task.get("load_repeated", False))}


def cohort_context(process_label: str | None) -> dict:
    """Context for cohort-wide results (distributions, domain and process comparisons,
    calibration, attribution): posture only. Task loads are defined for some tasks only and
    are provisional, so applying them across the cohort would make domain differences reflect
    which tasks have a load entry rather than the postures. They enter the expert comparison
    and a sensitivity analysis (topic/E/result/taskload_sensitivity)."""
    return {**task_context(process_label), "load_kg": 0.0, "load_repeated": False}
