# PWW-DISCLOSURE: public
"""
exp-002-standards-audit

Cell-by-cell audit of the reused legacy OWAS / RULA / REBA lookup
tables (ace_pose/ergo/*.py, ETRI ewhse code) against the clean-room
BIOERA v2 tables (ace_pose/ergo_v2/tables.py), which are transcribed
verbatim from the original standards specifications
(Karhu 1977; Louhevaara 1992; McAtamney 1993; Hignett 2000).

Verifies:
  - T01 (owas_wrong_cells) == 3
  - T02 (rula_tableA_wrong_cells_of_144) == 26
  - K1 / K2 / K15 (claim.md)
  - C0 (contribution)
  - E1 (evidence)

Records only facts; no pass/fail. /pw-verify judges.

Legacy tables are transcribed VERBATIM from the source lines cited
below; no interpretation. Provenance:
  - OWAS: ace_pose/ergo/owas.py:120-131 (function calcOWAS)
  - RULA: ace_pose/ergo/rula.py:223-258 (function calcRULA)
  - REBA: ace_pose/ergo/reba.py:182-208 (function calcREBA)

v2 tables imported directly from ace_pose.ergo_v2.tables.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TOPIC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TOPIC_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # type: ignore

from ace_pose.ergo_v2.tables import (  # type: ignore
    OWAS_AC as V2_OWAS,
    RULA_TABLE_A as V2_RULA_A,
    RULA_TABLE_B as V2_RULA_B,
    RULA_TABLE_C as V2_RULA_C,
    REBA_TABLE_A as V2_REBA_A,
    REBA_TABLE_B as V2_REBA_B,
    REBA_TABLE_C as V2_REBA_C,
)


# ------------------------------------------------------------------
# Legacy v1 tables transcribed VERBATIM (source lines cited in module docstring)
# ------------------------------------------------------------------

V1_OWAS = [
    [
        [(1,1,1),(1,1,1),(1,1,1),(2,2,2),(2,2,2),(1,1,1),(1,1,1)],
        [(1,1,1),(1,1,1),(1,1,1),(2,2,2),(2,2,2),(1,1,1),(1,1,1)],
        [(1,1,1),(1,1,1),(1,1,1),(2,2,3),(2,2,3),(1,1,1),(1,2,1)],
    ],
    [
        [(2,2,3),(2,2,3),(2,2,3),(3,3,3),(3,3,3),(2,2,2),(2,3,3)],
        [(2,2,3),(2,2,3),(2,3,3),(3,4,4),(3,4,4),(3,3,4),(2,3,4)],
        [(3,3,4),(2,2,3),(3,3,3),(3,4,4),(4,4,4),(4,4,4),(2,3,4)],
    ],
    [
        [(1,1,1),(1,1,1),(1,1,2),(3,3,3),(4,4,4),(1,1,1),(1,1,1)],
        [(2,2,3),(1,1,1),(1,1,2),(4,4,4),(4,4,4),(3,3,3),(1,1,1)],
        [(2,2,3),(1,1,1),(2,3,3),(4,4,4),(4,4,4),(4,4,4),(1,1,1)],
    ],
    [
        [(2,3,3),(2,2,3),(2,2,3),(4,4,4),(4,4,4),(4,4,4),(2,3,4)],
        [(3,3,4),(2,3,4),(3,3,4),(4,4,4),(4,4,4),(4,4,4),(2,3,4)],
        [(4,4,4),(2,3,4),(3,3,4),(4,4,4),(4,4,4),(4,4,4),(2,3,4)],
    ],
]

V1_RULA_A = [
    [[(1,2),(2,2),(2,3),(3,3)],[(2,2),(2,2),(3,3),(3,3)],[(2,3),(3,3),(3,3),(4,4)]],
    [[(2,3),(3,3),(3,4),(4,4)],[(3,3),(3,3),(3,4),(4,4)],[(3,4),(4,4),(4,4),(5,5)]],
    [[(3,3),(4,4),(4,4),(5,5)],[(3,4),(4,4),(4,4),(5,5)],[(4,4),(4,4),(4,5),(5,5)]],
    [[(4,4),(4,4),(4,5),(5,5)],[(4,4),(4,4),(4,5),(5,5)],[(4,4),(4,5),(5,5),(6,6)]],
    [[(5,5),(5,5),(5,6),(6,7)],[(5,6),(6,6),(6,7),(7,7)],[(6,6),(6,7),(7,7),(7,8)]],
    [[(7,7),(7,7),(7,8),(8,9)],[(8,8),(8,8),(8,9),(9,9)],[(9,9),(9,9),(9,9),(9,9)]],
]

V1_RULA_B = [
    [(1,3),(2,3),(3,4),(5,5),(6,6),(7,7)],
    [(2,3),(2,3),(4,5),(5,5),(6,7),(7,7)],
    [(3,3),(3,4),(4,5),(5,6),(6,7),(7,7)],
    [(5,5),(5,6),(6,7),(7,7),(7,7),(8,8)],
    [(7,7),(7,7),(7,8),(8,8),(8,8),(8,8)],
    [(8,8),(8,8),(8,8),(8,9),(9,9),(9,9)],
]

V1_RULA_C = [
    [1,2,3,3,4,5,5],
    [2,2,3,4,4,5,5],
    [3,3,3,4,4,5,6],
    [3,3,3,4,5,6,6],
    [4,4,4,5,6,7,7],
    [4,4,5,6,6,7,7],
    [5,5,6,6,7,7,7],
    [5,5,6,7,7,7,7],
]

V1_REBA_A = [
    [(1,2,3,4),(1,2,3,4),(3,3,5,6)],
    [(2,3,4,5),(3,4,5,6),(4,5,6,7)],
    [(2,4,5,6),(4,5,6,7),(5,6,7,8)],
    [(3,5,6,7),(5,6,7,8),(6,7,8,9)],
    [(4,6,7,8),(6,7,8,9),(7,8,9,9)],
]

V1_REBA_B = [
    [(1,2,2),(1,2,3)],
    [(1,2,3),(2,3,4)],
    [(3,4,5),(4,5,5)],
    [(4,5,5),(5,6,7)],
    [(6,7,8),(7,8,8)],
    [(7,8,8),(8,9,9)],
]

V1_REBA_C = [
    [1,1,1,2,3,3,4,5,6,7,7,7],
    [1,2,2,3,4,4,5,6,6,7,7,8],
    [2,3,3,3,4,5,6,7,7,8,8,8],
    [3,4,4,4,5,6,7,8,8,9,9,9],
    [4,4,4,5,6,7,8,8,9,9,9,9],
    [6,6,6,7,8,8,9,9,10,10,10,10],
    [7,7,7,8,9,9,9,10,10,11,11,11],
    [8,8,8,9,10,10,10,10,10,11,11,11],
    [9,9,9,10,10,10,11,11,11,12,12,12],
    [10,10,10,11,11,11,11,12,12,12,12,12],
    [11,11,11,11,12,12,12,12,12,12,12,12],
    [12,12,12,12,12,12,12,12,12,12,12,12],
]


# ------------------------------------------------------------------
# diff helpers
# ------------------------------------------------------------------

def to_np(lst):
    return np.array(lst)


def diff_owas():
    """v1 shape [back][arms][legs] with per-cell 3-tuple over loads.
    v2 shape (back, arms, legs, load) = (4, 3, 7, 3)."""
    diffs = []
    total = 0
    for b in range(4):
        for a in range(3):
            for l in range(7):
                for ld in range(3):
                    v1 = V1_OWAS[b][a][l][ld]
                    v2 = int(V2_OWAS[b, a, l, ld])
                    total += 1
                    if v1 != v2:
                        diffs.append({"axis": ["back", "arms", "legs", "load"],
                                      "code": [b + 1, a + 1, l + 1, ld + 1],
                                      "v1": v1, "v2": v2})
    return {"total_cells": total, "wrong_cells": len(diffs), "diffs": diffs}


def diff_rula_A():
    """v1 [upper][lower][wrist] tuple of (twist=1, twist=2). v2 shape (6,3,4,2)."""
    diffs = []
    total = 0
    for u in range(6):
        for lw in range(3):
            for w in range(4):
                for t in range(2):
                    v1 = V1_RULA_A[u][lw][w][t]
                    v2 = int(V2_RULA_A[u, lw, w, t])
                    total += 1
                    if v1 != v2:
                        diffs.append({"code": [u + 1, lw + 1, w + 1, t + 1],
                                      "v1": v1, "v2": v2})
    return {"total_cells": total, "wrong_cells": len(diffs), "diffs": diffs}


def diff_rula_B():
    """v1 [neck][trunk] tuple(legs1, legs2). v2 shape (6,6,2)."""
    diffs = []
    total = 0
    for n in range(6):
        for tr in range(6):
            for lg in range(2):
                v1 = V1_RULA_B[n][tr][lg]
                v2 = int(V2_RULA_B[n, tr, lg])
                total += 1
                if v1 != v2:
                    diffs.append({"code": [n + 1, tr + 1, lg + 1], "v1": v1, "v2": v2})
    return {"total_cells": total, "wrong_cells": len(diffs), "diffs": diffs}


def diff_rula_C():
    """v1 8x7, v2 shape (8,7)."""
    diffs = []; total = 0
    for r in range(8):
        for c in range(7):
            v1 = V1_RULA_C[r][c]; v2 = int(V2_RULA_C[r, c])
            total += 1
            if v1 != v2:
                diffs.append({"code": [r + 1, c + 1], "v1": v1, "v2": v2})
    return {"total_cells": total, "wrong_cells": len(diffs), "diffs": diffs}


def diff_reba_A():
    """v1 [trunk][neck] tuple(legs=1..4). v2 shape (5,3,4)."""
    diffs = []; total = 0
    for tr in range(5):
        for n in range(3):
            for lg in range(4):
                v1 = V1_REBA_A[tr][n][lg]
                v2 = int(V2_REBA_A[tr, n, lg])
                total += 1
                if v1 != v2:
                    diffs.append({"code": [tr + 1, n + 1, lg + 1], "v1": v1, "v2": v2})
    return {"total_cells": total, "wrong_cells": len(diffs), "diffs": diffs}


def diff_reba_B():
    """v1 [upper][lower] tuple(wrist=1..3). v2 shape (6,2,3)."""
    diffs = []; total = 0
    for u in range(6):
        for lw in range(2):
            for w in range(3):
                v1 = V1_REBA_B[u][lw][w]
                v2 = int(V2_REBA_B[u, lw, w])
                total += 1
                if v1 != v2:
                    diffs.append({"code": [u + 1, lw + 1, w + 1], "v1": v1, "v2": v2})
    return {"total_cells": total, "wrong_cells": len(diffs), "diffs": diffs}


def diff_reba_C():
    diffs = []; total = 0
    for r in range(12):
        for c in range(12):
            v1 = V1_REBA_C[r][c]; v2 = int(V2_REBA_C[r, c])
            total += 1
            if v1 != v2:
                diffs.append({"code": [r + 1, c + 1], "v1": v1, "v2": v2})
    return {"total_cells": total, "wrong_cells": len(diffs), "diffs": diffs}


def main():
    audit = {
        "OWAS": diff_owas(),
        "RULA_A": diff_rula_A(),
        "RULA_B": diff_rula_B(),
        "RULA_C": diff_rula_C(),
        "REBA_A": diff_reba_A(),
        "REBA_B": diff_reba_B(),
        "REBA_C": diff_reba_C(),
    }
    total_wrong = sum(t["wrong_cells"] for t in audit.values())
    total_cells = sum(t["total_cells"] for t in audit.values())

    metrics = {
        # targets.md T01, T02
        "T01_owas_wrong_cells": audit["OWAS"]["wrong_cells"],
        "T02_rula_tableA_wrong_cells_of_144": audit["RULA_A"]["wrong_cells"],
        "T02_rula_tableA_total_cells": audit["RULA_A"]["total_cells"],
        # additional per-table counts
        "rula_tableB_wrong_cells": audit["RULA_B"]["wrong_cells"],
        "rula_tableC_wrong_cells": audit["RULA_C"]["wrong_cells"],
        "reba_tableA_wrong_cells": audit["REBA_A"]["wrong_cells"],
        "reba_tableB_wrong_cells": audit["REBA_B"]["wrong_cells"],
        "reba_tableC_wrong_cells": audit["REBA_C"]["wrong_cells"],
        "total_wrong_cells": total_wrong,
        "total_cells_audited": total_cells,
        "v2_source": "ace_pose/ergo_v2/tables.py (verbatim from standards specs)",
        "v1_source": "ace_pose/ergo/{owas,rula,reba}.py (ETRI ewhse legacy)",
        "reference_standards": [
            "karhu1977owas",
            "louhevaara1992owas",
            "mcatamney1993rula",
            "mcatamney2004rula",
            "hignett2000reba",
        ],
    }

    result_dir = Path(os.environ.get("PWW_RESULT_DIR", TOPIC_ROOT / "result" / "exp-002-standards-audit"))
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (result_dir / "raw.json").write_text(json.dumps({"metrics": metrics, "audit": audit}, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
