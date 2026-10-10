"""Canonical lookup tables for OWAS, RULA, and REBA.

All values are transcribed VERBATIM from the documented standards
(`ace_pose/standards/{owas,rula,reba}.md`) and cross-validated against
at least two independent sources (Karhu 1977 / Louhevaara 1992;
McAtamney 1993 / Cornell PDF / ErgoPlus; Hignett 2000 / Cornell /
rs9000 reference implementation).

Do not edit these in place. If a value here disagrees with a source,
fix it AND add a comment citing both sources.
"""
from __future__ import annotations

import numpy as np


# ============================================================
# OWAS — Action Category lookup
# ============================================================
# Indexed as OWAS_AC[back-1, arms-1, legs-1, load-1].
# Shape (4, 3, 7, 3); values ∈ {1, 2, 3, 4}.
# Source: Louhevaara & Suurnäkki 1992, cross-checked with Ergonautas and
# `rs9000/ergonomics`. See `standards/owas.md §3`.
OWAS_AC = np.array([
    # Back = 1 (Straight)
    [
        [[1,1,1],[1,1,1],[1,1,1],[2,2,2],[2,2,2],[1,1,1],[1,1,1]],  # Arms = 1
        [[1,1,1],[1,1,1],[1,1,1],[2,2,2],[2,2,2],[1,1,1],[1,1,1]],  # Arms = 2
        [[1,1,1],[1,1,1],[1,1,1],[2,2,3],[2,2,3],[1,1,1],[1,1,2]],  # Arms = 3
    ],
    # Back = 2 (Bent)
    [
        [[2,2,3],[2,2,3],[2,2,3],[3,3,3],[3,3,3],[2,2,2],[2,3,3]],  # Arms = 1
        [[2,2,3],[2,2,3],[2,3,3],[3,4,4],[3,4,4],[3,3,4],[2,3,4]],  # Arms = 2
        [[3,3,4],[2,2,3],[3,3,3],[3,4,4],[4,4,4],[4,4,4],[2,3,4]],  # Arms = 3
    ],
    # Back = 3 (Twisted)
    [
        [[1,1,1],[1,1,1],[1,1,2],[3,3,3],[4,4,4],[1,1,1],[1,1,1]],  # Arms = 1
        [[2,2,3],[1,1,1],[1,1,2],[4,4,4],[4,4,4],[3,3,3],[1,1,1]],  # Arms = 2
        [[2,2,3],[1,1,1],[2,3,3],[4,4,4],[4,4,4],[4,4,4],[1,1,2]],  # Arms = 3
    ],
    # Back = 4 (Bent AND Twisted)
    [
        [[2,3,3],[2,2,3],[2,3,3],[4,4,4],[4,4,4],[4,4,4],[2,3,4]],  # Arms = 1
        [[3,3,4],[2,3,4],[3,3,4],[4,4,4],[4,4,4],[4,4,4],[2,3,4]],  # Arms = 2
        [[4,4,4],[2,3,4],[3,3,4],[4,4,4],[4,4,4],[4,4,4],[2,3,4]],  # Arms = 3
    ],
], dtype=np.int8)


# ============================================================
# RULA — Table A (Arm + Wrist)
# ============================================================
# Indexed as RULA_TABLE_A[upper-1, lower-1, wrist-1, twist-1].
# Shape (6, 3, 4, 2); values ∈ {1..9}.
# Source: McAtamney & Corlett 1993, Applied Ergonomics 24(2):91-99,
# Table 1 (page 95). Re-transcribed verbatim from the original paper
# PDF on 2026-08-11 after discovering that the previous transcription
# (from the Cornell/Hedge worksheet) diverged from the primary source
# at 31 of 144 cells. See `standards/rula.md §4` and the correction
# note in `paper/preregistration/X4-hashes-2026-08-10.txt` amendment.
RULA_TABLE_A = np.array([
    # Upper Arm = 1
    [
        [[1, 2], [2, 2], [2, 3], [3, 3]],  # Lower = 1
        [[2, 2], [2, 2], [3, 3], [3, 3]],  # Lower = 2
        [[2, 3], [3, 3], [3, 3], [4, 4]],  # Lower = 3
    ],
    # Upper Arm = 2
    [
        [[2, 3], [3, 3], [3, 4], [4, 4]],  # Lower = 1
        [[3, 3], [3, 3], [3, 4], [4, 4]],  # Lower = 2
        [[3, 4], [4, 4], [4, 4], [5, 5]],  # Lower = 3
    ],
    # Upper Arm = 3
    [
        [[3, 3], [4, 4], [4, 4], [5, 5]],  # Lower = 1
        [[3, 4], [4, 4], [4, 4], [5, 5]],  # Lower = 2
        [[4, 4], [4, 4], [4, 5], [5, 5]],  # Lower = 3
    ],
    # Upper Arm = 4
    [
        [[4, 4], [4, 4], [4, 5], [5, 5]],  # Lower = 1
        [[4, 4], [4, 4], [4, 5], [5, 5]],  # Lower = 2
        [[4, 4], [4, 5], [5, 5], [6, 6]],  # Lower = 3
    ],
    # Upper Arm = 5
    [
        [[5, 5], [5, 5], [5, 6], [6, 7]],  # Lower = 1
        [[5, 6], [6, 6], [6, 7], [7, 7]],  # Lower = 2
        [[6, 6], [6, 7], [7, 7], [7, 8]],  # Lower = 3
    ],
    # Upper Arm = 6
    [
        [[7, 7], [7, 7], [7, 8], [8, 9]],  # Lower = 1
        [[8, 8], [8, 8], [8, 9], [9, 9]],  # Lower = 2
        [[9, 9], [9, 9], [9, 9], [9, 9]],  # Lower = 3
    ],
], dtype=np.int8)


# ============================================================
# RULA — Table B (Neck + Trunk + Legs)
# ============================================================
# Indexed as RULA_TABLE_B[neck-1, trunk-1, legs-1].
# Shape (6, 6, 2); values ∈ {1..9}.
# Source: McAtamney & Corlett 1993; Cornell worksheet.
# See `standards/rula.md §5`.
RULA_TABLE_B = np.array([
    # Neck = 1
    [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
    # Neck = 2
    [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
    # Neck = 3
    [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
    # Neck = 4
    [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
    # Neck = 5
    [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
    # Neck = 6
    [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
], dtype=np.int8)


# ============================================================
# RULA — Table C (Grand Score)
# ============================================================
# Indexed as RULA_TABLE_C[wrist_arm-1, neck_trunk_leg-1].
# Wrist/Arm score is clamped to 8; Neck/Trunk/Leg clamped to 7.
# Shape (8, 7); values ∈ {1..7}.
# Source: McAtamney & Corlett 1993; Cornell worksheet.
# See `standards/rula.md §7`.
RULA_TABLE_C = np.array([
    [1, 2, 3, 3, 4, 5, 5],  # Wrist/Arm = 1
    [2, 2, 3, 4, 4, 5, 5],  # Wrist/Arm = 2
    [3, 3, 3, 4, 4, 5, 6],  # Wrist/Arm = 3
    [3, 3, 3, 4, 5, 6, 6],  # Wrist/Arm = 4
    [4, 4, 4, 5, 6, 7, 7],  # Wrist/Arm = 5
    [4, 4, 5, 6, 6, 7, 7],  # Wrist/Arm = 6
    [5, 5, 6, 6, 7, 7, 7],  # Wrist/Arm = 7
    [5, 5, 6, 7, 7, 7, 7],  # Wrist/Arm = 8+
], dtype=np.int8)


# ============================================================
# REBA — Table A (Trunk + Neck + Legs)
# ============================================================
# Indexed as REBA_TABLE_A[trunk-1, neck-1, legs-1].
# Shape (5, 3, 4); values ∈ {1..9}.
# Source: Hignett & McAtamney 2000; Cornell PDF; rs9000.
# See `standards/reba.md §4`.
REBA_TABLE_A = np.array([
    # Trunk = 1
    [
        [1, 2, 3, 4],   # Neck = 1
        [1, 2, 3, 4],   # Neck = 2
        [3, 3, 5, 6],   # Neck = 3
    ],
    # Trunk = 2
    [
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ],
    # Trunk = 3
    [
        [2, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
    ],
    # Trunk = 4
    [
        [3, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ],
    # Trunk = 5
    [
        [4, 6, 7, 8],
        [6, 7, 8, 9],
        [7, 8, 9, 9],
    ],
], dtype=np.int8)


# ============================================================
# REBA — Table B (Upper arm + Lower arm + Wrist)
# ============================================================
# Indexed as REBA_TABLE_B[upper-1, lower-1, wrist-1].
# Shape (6, 2, 3); values ∈ {1..9}.
# Source: Hignett & McAtamney 2000.
# See `standards/reba.md §6`.
REBA_TABLE_B = np.array([
    [[1, 2, 2], [1, 2, 3]],   # Upper = 1
    [[1, 2, 3], [2, 3, 4]],   # Upper = 2
    [[3, 4, 5], [4, 5, 5]],   # Upper = 3
    [[4, 5, 5], [5, 6, 7]],   # Upper = 4
    [[6, 7, 8], [7, 8, 8]],   # Upper = 5
    [[7, 8, 8], [8, 9, 9]],   # Upper = 6
], dtype=np.int8)


# ============================================================
# REBA — Table C (Final A + B combination)
# ============================================================
# Indexed as REBA_TABLE_C[score_a-1, score_b-1].
# NOT symmetric: TABLE_C[A][B] != TABLE_C[B][A].
# Shape (12, 12); values ∈ {1..12}.
# Source: Hignett & McAtamney 2000; rs9000 numpy verification.
# See `standards/reba.md §8`.
REBA_TABLE_C = np.array([
    [ 1,  1,  1,  2,  3,  3,  4,  5,  6,  7,  7,  7],   # Score A = 1
    [ 1,  2,  2,  3,  4,  4,  5,  6,  6,  7,  7,  8],   # Score A = 2
    [ 2,  3,  3,  3,  4,  5,  6,  7,  7,  8,  8,  8],   # Score A = 3
    [ 3,  4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9],   # Score A = 4
    [ 4,  4,  4,  5,  6,  7,  8,  8,  9,  9,  9,  9],   # Score A = 5
    [ 6,  6,  6,  7,  8,  8,  9,  9, 10, 10, 10, 10],   # Score A = 6
    [ 7,  7,  7,  8,  9,  9,  9, 10, 10, 11, 11, 11],   # Score A = 7
    [ 8,  8,  8,  9, 10, 10, 10, 10, 10, 11, 11, 11],   # Score A = 8
    [ 9,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12, 12],   # Score A = 9
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],   # Score A = 10
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],   # Score A = 11
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],   # Score A = 12
], dtype=np.int8)


# Sanity checks at import time.
assert OWAS_AC.shape == (4, 3, 7, 3), OWAS_AC.shape
assert RULA_TABLE_A.shape == (6, 3, 4, 2), RULA_TABLE_A.shape
assert RULA_TABLE_B.shape == (6, 6, 2), RULA_TABLE_B.shape
assert RULA_TABLE_C.shape == (8, 7), RULA_TABLE_C.shape
assert REBA_TABLE_A.shape == (5, 3, 4), REBA_TABLE_A.shape
assert REBA_TABLE_B.shape == (6, 2, 3), REBA_TABLE_B.shape
assert REBA_TABLE_C.shape == (12, 12), REBA_TABLE_C.shape

assert OWAS_AC.min() >= 1 and OWAS_AC.max() <= 4
assert RULA_TABLE_A.min() >= 1 and RULA_TABLE_A.max() <= 9
assert RULA_TABLE_B.min() >= 1 and RULA_TABLE_B.max() <= 9
assert RULA_TABLE_C.min() >= 1 and RULA_TABLE_C.max() <= 7
assert REBA_TABLE_A.min() >= 1 and REBA_TABLE_A.max() <= 9
assert REBA_TABLE_B.min() >= 1 and REBA_TABLE_B.max() <= 9
assert REBA_TABLE_C.min() >= 1 and REBA_TABLE_C.max() <= 12
