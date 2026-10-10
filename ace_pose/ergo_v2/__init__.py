"""ergo_v2 — clean re-implementation of OWAS / RULA / REBA from the
documented standards (see `ace_pose/standards/{owas,rula,reba}.md`).

Differs from the legacy `ace_pose.ergo` package by:
  - lookup tables transcribed verbatim from cross-validated authoritative
    sources (the legacy code has 3 wrong OWAS cells and 26 wrong RULA
    Table-A cells; see `standards/audit.md`)
  - canonical angle thresholds (no bias-shifted boundaries)
  - structured input: dict-of-joints instead of indexed measurement list
  - unified `ErgoResult` output schema across all three methods
  - explicit handling of missing finger keypoints (configurable defaults)
  - silent by default; opt-in `logging`
"""
from .result import ErgoResult, OwasResult, RulaResult, RebaResult
from .owas import score_owas
from .rula import score_rula
from .reba import score_reba
from .pipeline import score_all

__all__ = [
    "ErgoResult",
    "OwasResult", "RulaResult", "RebaResult",
    "score_owas", "score_rula", "score_reba",
    "score_all",
]
