# PWW-DISCLOSURE: public
"""exp-005 per-domain ECE breakdown for T03 (RULA ECE reduction) and
T04 (REBA ECE reduction).

Fact-only. Success/failure is /pw-verify prerogative.

Input: ace_pose/_runs/batch_v1/c4_calibration_report.csv (produced by
the frozen C4 calibration pipeline).

Output: PWW_RESULT_DIR/metrics.json with per-(domain, method) ECE
before, ECE after, absolute delta, and relative delta.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def resolve_input() -> Path:
    from pww_data import resolve  # type: ignore

    return Path(resolve("c4_calibration_report"))


def _fallback_input() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "ace_pose" / "_runs" / "batch_v1" / "c4_calibration_report.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("c4_calibration_report.csv not found on ancestor path")


def load_report(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def compile_metrics(rows: list[dict]) -> dict:
    per_row = []
    for row in rows:
        n = int(row["n"])
        ece_before = float(row["ece_before"])
        ece_after = float(row["ece_after"])
        delta_abs = ece_before - ece_after
        delta_rel = delta_abs / ece_before if ece_before > 0 else float("nan")
        per_row.append(
            {
                "domain": row["domain"],
                "method": row["method"],
                "n": n,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "ece_delta_abs": delta_abs,
                "ece_delta_rel": delta_rel,
                "temperature": float(row["temperature"]),
            }
        )

    def _weighted(method: str) -> dict:
        subset = [r for r in per_row if r["method"] == method]
        n_total = sum(r["n"] for r in subset)
        if n_total == 0:
            return {"n": 0}
        before = sum(r["ece_before"] * r["n"] for r in subset) / n_total
        after = sum(r["ece_after"] * r["n"] for r in subset) / n_total
        return {
            "n": n_total,
            "ece_before_weighted": before,
            "ece_after_weighted": after,
            "ece_delta_abs_weighted": before - after,
            "ece_delta_rel_weighted": (before - after) / before if before > 0 else float("nan"),
        }

    return {
        "per_row": per_row,
        "aggregates": {
            "owas": _weighted("owas"),
            "rula": _weighted("rula"),
            "reba": _weighted("reba"),
        },
    }


def main() -> None:
    try:
        input_path = resolve_input()
    except Exception:
        input_path = _fallback_input()

    rows = load_report(input_path)
    metrics = compile_metrics(rows)
    metrics["input_path_basename"] = input_path.name

    out_dir = Path(os.environ.get("PWW_RESULT_DIR", ".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
