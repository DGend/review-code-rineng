# Audit: Existing Certified Ergo Code vs Documented Standards

Cross-checked the EWHSE certified ergo modules (`ergo/owas.py`, `ergo/rula.py`,
`ergo/reba.py`, `ergo/config.py`) against the three standard specifications in
`standards/{owas,rula,reba}.md`. Findings are ordered by severity (highest first).
Severity legend:

- **CRITICAL** — produces wrong action category / risk score for plausible inputs.
- **MAJOR** — diverges from canonical reference but stays within ±1 score band most of the time.
- **MINOR** — code style / default behaviour that under-reports without changing
  the structural classification.

---

## A. OWAS

### A.1 Action-category lookup table — 3 mismatched cells (out of 252)
**Severity: CRITICAL.**
Direct comparison of `ergo/owas.py:OWAS_TABLE` against the cross-validated
252-entry matrix in `standards/owas.md §3`.

| Back | Arms | Legs | Code value | Standard value |
|------|------|------|-----------|----------------|
| 1    | 3    | 7    | `(1, 2, 1)` | `(1, 1, 2)` |
| 3    | 3    | 7    | `(1, 1, 1)` | `(1, 1, 2)` |
| 4    | 1    | 3    | `(2, 2, 3)` | `(2, 3, 3)` |

The first two cells (Legs=7 = walking) flip Load=2 and Load=3 outcomes; the
third cell drops a load-1 entry by one point. All three cells lie on the
load axis, which suggests a transcription error in the original Korean
reference at the time of EWHSE coding.

**Fix:** replace with the Python literal in `standards/owas.md §3` (the
verbatim Ergonautas / Louhevaara matrix).

### A.2 Knee straight-vs-bent threshold uses 160°, canonical is 150°
**Severity: MAJOR.**
`config.py:OWAS_LOWER_ANGLE = 160.0` — but the Karhu/Louhevaara canonical
threshold for distinguishing a "straight" knee from a "bent" knee is
**150°** (thigh–calf angle ≥ 150° → straight).

Effect: any worker with knee flexion between 20° and 30° is classified as
"bent" by our code (code 4/5) when the standard would classify them as
"straight" (code 2/3). In factory poses this is a frequent source of
risk over-reporting.

**Fix:** set `OWAS_LOWER_ANGLE = 150.0` to match canonical.

### A.3 Back-code logic: lateral bend + sagittal flexion does not yield Back=4
**Severity: MAJOR.**
`owas.py:194–220` orders the back-code branches as:

```python
if nTmpVal < 20:               # not bent
    if side_bend >= 20: nCodeOwasBack = 3   # twisted via lateral
if nTmpVal >= 20:               # bent
    nCodeOwasBack = 2                       # overwrites!
if twist >= 20 and code != 3:
    nCodeOwasBack += 2
```

The lateral-bend test sits inside the *not bent* branch and is **never
evaluated** in the bent path. Standard rule (`standards/owas.md §2.1`)
says **Back = 4 ⇔ sagittal flex > 20° AND (rotation OR lateral) > 20°**.

| Input combination                  | Standard | Code | Drift |
|------------------------------------|---------:|-----:|------:|
| bent + axial-twist + lateral       | 4 | 4 | 0 |
| bent + axial-twist + no lateral    | 4 | 4 | 0 |
| **bent + no axial-twist + lateral**| **4** | **2** | **−2** |
| not bent + axial-twist + lateral   | 3 | 3 | 0 |
| not bent + axial only              | 3 | 3 | 0 |
| not bent + lateral only            | 3 | 3 | 0 |

A worker who bends forward and side-bends (e.g., reaching across the body
to pick from a low conveyor) drops from Back=4 to Back=2 → action
category shifts from "act NOW" to "act in near future".

**Fix:** combine the lateral and axial test into a unified `twisted` flag
before the bent check:
```python
bent     = sagittal_flex > 20
twisted  = (axial_twist > 20) or (lateral_bend > 20)
back = 1 + int(bent) + 2*int(twisted)        # 1, 2, 3, or 4
```

### A.4 Sitting detection misclassifies deep squats
**Severity: MAJOR.**
`owas.py:305–306`:
```python
elif nTmpVal < 90 and nTmpVal1 < 90:
    nCodeOwasLower = 1   # "sitting"
```

A worker squatting deeply (knee flexion > 90°, e.g., welding at floor
level) has both `nTmpVal` (knee angle) under 90°, and is therefore
classified as **sitting (code 1) instead of both-knees-bent (code 4)**.
Sitting is rated 1 = "no harmful effect"; bent is 2-3. This systematically
under-reports the risk of low-stance assembly / welding / picking tasks.

Per `standards/owas.md §2.3`, sitting is detected from **hip flexion ≈
90° together with both knees flexed AND pelvis low / supported**, not
from knee angle alone. The Ergonautas reference adds an explicit
`hip_seat_height < knee_height` check.

**Fix:** require hip-flexion + low pelvis to classify as sitting; otherwise
fall through to the bent-knees branch.

### A.5 Walking only via external flag
**Severity: MINOR.**
`owas.py:350` sets `legs = 7` only when the caller passes
`code_walking = 1`. No auto-detection from a frame sequence. Acceptable
for single-frame inference but should be documented as a known
limitation.

### A.6 Leg-decision branches use overlapping OR/AND conditions
**Severity: MINOR.**
`owas.py:295–306` mixes `nTmpVal > 160 OR nTmpVal1 between 90 and 160` etc.
The ladder is hard to reason about and has at least one provably
unreachable branch (after lines 295–298 fire, lines 299–304 already exclude
the remaining cases for the canonical 150° threshold). Rewrite as the
clean priority table in `standards/owas.md §4.3`.

---

## B. RULA

### B.1 Table A — multiple off-by-one errors (high cell count)
**Severity: CRITICAL.**
Comparing `rula.py:RULA_TABLE_A` against the verified Cornell/ErgoPlus
Table A (`standards/rula.md §4`). Each row is laid out as
`(W1T1, W1T2, W2T1, W2T2, W3T1, W3T2, W4T1, W4T2)`.

| Upper | Lower | Code values                  | Standard values              | Mismatches |
|-------|-------|------------------------------|------------------------------|------------|
| 1     | 1     | 1 2 2 2 2 3 3 3              | 1 2 2 2 2 3 3 3              | 0          |
| 1     | 2     | 2 2 2 2 3 3 3 3              | 2 2 2 2 3 3 3 3              | 0          |
| 1     | 3     | 2 3 **3** 3 3 3 4 4          | 2 3 **2** 3 3 3 4 4          | 1 (W2T1)   |
| 2     | 1     | 2 **3** **3** 3 3 **4** 4 4  | 2 **2** **2** 3 3 **3** 4 4  | 3          |
| 2     | 2     | **3 3 3** 3 3 **4** 4 4      | **2 2 2** 3 3 **3** 4 4      | 4          |
| 2     | 3     | 2 3 3 3 3 4 4 5              | 2 3 3 3 3 4 4 5              | 0          |
| 3     | 1     | **3 3** **4 4** **4 4** 5 5  | **2 3** **3 3** **4 4** 5 5  | 4          |
| 3     | 2     | **3 4** **4 4 4 4** 5 5      | **2 3** **3 3 4 4** 5 5      | 5          |
| 3     | 3     | **4 4** **4 4** 4 **5** 5 5  | **2 3** **3 4** 4 **4** 5 5  | 5          |
| 4     | 1     | **4** 4 4 4 4 4 5 5          | **3** 4 4 4 4 4 5 5          | 1          |
| 4     | 2     | **4** 4 4 4 4 4 5 5          | **3** 4 4 4 4 4 5 5          | 1          |
| 4     | 3     | **4** 4 4 5 5 5 6 6          | **3** 4 4 5 5 5 6 6          | 1          |
| 5     | 1     | 5 5 5 5 5 6 6 7              | 5 5 5 5 5 6 6 7              | 0          |
| 5     | 2     | 5 6 6 6 6 7 7 7              | 5 6 6 6 6 7 7 7              | 0          |
| 5     | 3     | 6 6 6 7 7 7 7 8              | 6 6 6 7 7 7 7 8              | 0          |
| 6     | 1     | 7 7 7 7 7 8 8 9              | 7 7 7 7 7 8 8 9              | 0          |
| 6     | 2     | **8** 8 8 8 8 9 9 9          | **7** 8 8 8 8 9 9 9          | 1          |
| 6     | 3     | 9 9 9 9 9 9 9 9              | 9 9 9 9 9 9 9 9              | 0          |

Total: **26 mismatched cells out of 144 (18 %)**. The mismatches are
concentrated in upper=2-4, where the code is consistently 1 score
**higher** than the standard. This pattern suggests a wholesale row-shift
in the source from which the Korean team transcribed Table A.

**Effect on grand score:** Table-A entry feeds the "Wrist & Arm" row index
into Table C. A row-shift of +1 propagates to roughly +0–1 of final RULA
grand score for upper-arm scores 2–4, which is the most common
upper-arm range in factory work (arms slightly flexed at 20°–90°).

**Fix:** replace `RULA_TABLE_A` with the verbatim 144-cell map from
`standards/rula.md §4`.

### B.2 Wrist twist defaulted to 1 with a "TODO" comment in production
**Severity: MAJOR.**
`rula.py:494` —
```python
nCodeWristTwist = 1   # default
...
#       nCodeWristTwist = 2
        print("w.F")
        # 추후 처리.
        pass
```

The wrist-twist sub-score is hard-coded to 1 with code that mutates it to
2 commented out and a Korean comment "to be handled later". This is the
**joint-author-acknowledged gap** that motivates our paper's hand-aware
contribution. Re-implementation should expose `wrist_twist_default = 1`
as a parameter and add a deviation rule when hand keypoints are
available.

### B.3 Trunk / neck / upper-arm thresholds deviate from canonical
**Severity: MAJOR.**

| Config constant            | Code value                          | Canonical             | Issue |
|----------------------------|-------------------------------------|-----------------------|-------|
| `RULA_TRUNK_ANGLE`         | `(-20, -18, 8, 19, 56)` — 5 thresholds | `0, 20, 60` (3 thresholds) | Wrong shape; appears to mix custom flex/ext boundaries |
| `RULA_NECK_ANGLE`          | `(-5, 20, 35)`                      | `0, 10, 20`           | Score-2 threshold 10° is missed; 35° boundary spurious |
| `RULA_UPPER_ARM_ANGLE`     | `(-20, 20, 45, 100)`                | `(-20, 20, 45, 90)`   | Score-4 boundary at 100° instead of 90° |
| `RULA_WRIST`               | `(25, 15, 10)`                      | `0, 15` (2 thresholds) | Wrong shape |

The trunk constants in particular are suspect: -20° / -18° / 8° / 19° / 56°
is not a clean RULA boundary; it looks like a per-axis offset injected to
compensate for the legacy 27-point model's calibration bias. In a
clean-room re-implementation against RTMW3D 133-point input we should
drop these offsets and use the canonical 0° / 20° / 60° boundaries.

### B.4 Tables B and C match canonical reference
**Severity: PASS.**
`RULA_TABLE_B` (6 × 6 × 2 = 72 cells) and `RULA_TABLE_C` (8 × 7 = 56
cells) match `standards/rula.md §5 and §7` exactly. No action.

### B.5 Lower-arm score for "out-to-side" used as +1 may double-count
**Severity: MINOR.**
The canonical "lower arm crosses midline" (+1) and "lower arm works out
to side" (+1) are mutually exclusive in most postures but the existing
code applies them independently. For a worker reaching across to the far
side, both flags can fire simultaneously, producing a lower-arm score
that hits the clamp at 3 too aggressively. Verify against the original
McAtamney & Corlett 1993 worksheet (which lists these as alternatives
sharing one +1 adjustment).

---

## C. REBA

### C.1 Tables A, B, C all match canonical reference
**Severity: PASS.**

| Table | Shape | Cells matched | Notes |
|-------|------:|---------------|-------|
| A     | 5 × 3 × 4 = 60 | 60 / 60 ✓ | Indexed `[trunk-1][neck-1] → (legs1, legs2, legs3, legs4)`, equivalent to standard's `[neck-1][trunk-1][legs-1]` |
| B     | 6 × 2 × 3 = 36 | 36 / 36 ✓ | Matches Cornell / Ergonautas |
| C     | 12 × 12 = 144  | 144 / 144 ✓ | The asymmetric matrix is reproduced correctly |

No action.

### C.2 Activity Score parameters are wired but default-disabled
**Severity: MAJOR.**
`reba.py:227–229`:
```python
nParamActivityScore  = False   # held > 1 minute
nParamActivityScore1 = False   # repetitive > 4×/min
nParamActivityScore2 = False   # unstable / rapid changes
```

These are **never mutated** anywhere in the function. Lines 550–552
correctly sum them into `nScoreActivity`, and line 578 correctly adds
`nScoreActivity` to the Table-C output. But because the upstream flags
are always False, the activity contribution is always **0** in
production runs.

The REBA standard considers the activity score **non-optional**
(`standards/reba.md §9, §13.1`). Forgetting it under-reports the final
REBA score by **1–3 points**, frequently dropping action level from
"High / act soon" down to "Medium / act when convenient" — the single
most common reason published REBA pipelines under-report risk.

**Fix:** either
- accept the three booleans as function parameters (caller supplies from
  task metadata or temporal analysis), or
- default the static-posture flag to **True** for held single-frame
  inference (a conservative choice that matches the typical use case),
  or
- compute the flags from a short temporal window when a frame sequence
  is available.

### C.3 Coupling defaults to 0 (Good)
**Severity: MAJOR.**
`reba.py:226` initialises `nParamCoupling = 0` and never updates it.
Score B gets no coupling penalty. The REBA standard
(`standards/reba.md §13.2`) flags "defaulting coupling to 0" as the
second-most common implementation error after the activity score.

**Recommendation:** default to **1 (Fair)** as the conservative middle
ground, with the parameter surfaced to the caller. For our manufacturing
+ eldercare validation set, a per-process coupling table would be even
better (welding torch = poor, hand-tools = fair, smooth handles = good).

### C.4 Trunk and neck thresholds deviate from canonical
**Severity: MAJOR.**
| Config constant            | Code value          | Canonical    | Issue |
|----------------------------|---------------------|--------------|-------|
| `REBA_TRUNK_ANGLE`         | `(-20, 8, 35, 60)`  | `(0, 20, 60)`| Wrong shape; spurious 8° and 35° |
| `REBA_NECK_ANGLE`          | `(-5, 15)`          | `(0, 20)`    | Off by ~5° in both boundaries |
| `REBA_UPPER_ARM_ANGLE`     | `(-20, 20, 45, 90)` | `(-20, 20, 45, 90)` | ✓ |
| `REBA_LOWER_ARM_ANGLE`     | `(60, 100)`         | `(60, 100)`  | ✓ |
| `REBA_WRIST`               | `15`                | `15`         | ✓ |
| `REBA_LEG`                 | `(30, 60)`          | `(30, 60)`   | ✓ |

Same pattern as RULA: the trunk and neck axes use bias-adjusted angles.
Re-implementation should use canonical thresholds.

### C.5 Knee-flexion adjustment applied unconditionally
**Severity: MINOR.**
Per `standards/reba.md §3.3`, the knee-flexion +1/+2 adjustment is
explicitly **not applied when sitting**. The existing code applies it in
all cases. Add the sitting-exclusion guard.

---

## D. Cross-cutting findings

### D.1 Input keypoint format is the project's 27-joint EWHSE layout
All three modules read indexed measurements from `nJsonRulaAngle[i]`,
`nJsonOwasAngle[i]`, etc. The index map is implicit and brittle.
Re-implementation should accept structured joint dictionaries
(`{shoulder_l: (x,y,z), shoulder_r: ..., ...}`) — far easier to map
from the RTMW3D 133-pt output via the major-25 filter
(`ace_pose/major_joints.py`).

### D.2 Stateful prints, no logging
All three modules write directly to stdout (`print(">>> Reporting OWAS")`
etc.). Make them silent by default; expose a `debug=True` parameter that
routes through Python's `logging` module. This is a sanity precondition
for running the modules in a 6,007-image batch without filling the
terminal.

### D.3 Returns differ by method (tuple vs structured dict)
- OWAS returns `(back, upper, lower, load, action_category)`.
- RULA returns `(analysis, params, codes, scores)`.
- REBA returns `(analysis, params, codes, scores)`.

Make all three return the same `ErgoResult` dataclass with `.score`,
`.action_level`, `.sub_scores`, `.flags`. Single output schema makes the
downstream calibration / MC pipeline simpler.

### D.4 No unit tests; no golden reference cases
None of the modules has a unit test. A standard QA practice for ergo
re-implementations is to encode the worked examples from the
McAtamney & Corlett 1993, Hignett & McAtamney 2000, and Karhu et al.
1977 papers as golden tests so any future refactor that breaks Table A
is caught at CI time. Add these once the re-implementation is in place.

---

## E. Suggested re-implementation plan

1. **Create `ace_pose/ergo_v2/` package** that mirrors `ergo/` but is
   built directly from the documented standards. Keep `ergo/` untouched
   so we can A/B-compare per-image scores.
2. **Implement the three lookup tables verbatim** from
   `standards/{owas,rula,reba}.md §3-§7`. Cells, indexing, and clamps
   come from the spec docs — not from the existing code.
3. **Compute joint angles from the major-25 layout** in
   `ace_pose/major_joints.py`. Body reference frame, sagittal /
   coronal / transverse projections, all formulas per
   `standards/{rula,reba}.md §9` and `standards/owas.md §4`.
4. **Default missing-input handling explicitly:** wrist subscore = 2 when
   finger keypoints absent; wrist twist = 1; coupling = 1 (fair);
   activity-static = True for single-frame mode. Surface every default
   as a parameter.
5. **Single `ErgoResult` dataclass** carries the multi-method outputs
   plus per-sub-score traceability (which joint vectors fed each
   angle, which thresholds fired). This becomes the input to the C3
   confidence-MC layer and the C4 domain-calibration layer.
6. **Encode golden test cases** from the three seminal papers' worked
   examples as `pytest` fixtures. Each must reproduce the published
   score to ±0.

The audit findings catalogued above become a paper figure / appendix
table: "Errata in legacy automated ergo implementations and their
clinical consequences."

---

## F. Summary

| Module | Tables OK? | Logic bugs | Threshold drift | Action |
|--------|-----------|------------|-----------------|--------|
| OWAS   | 3/252 cells wrong | Back-4 lateral; squat-as-sitting | knee 160° vs 150° | Re-implement |
| RULA   | A: 26/144 wrong; B, C ✓ | Wrist twist hard-coded to 1 | Trunk/neck/upper-arm | Re-implement |
| REBA   | All 3 tables ✓ | Activity score always 0; coupling 0 | Trunk/neck | Re-implement |

**Recommendation:** proceed to Task #20 — build the major-joint subset
filter (already drafted in `ace_pose/major_joints.py`) — then write
`ergo_v2/` as a clean implementation against the documented standards.
Run both the legacy and the re-implemented pipelines on the 6,007-image
test set and report the per-method delta in the paper as part of the
"Modernized automatic ergonomic assessment" contribution (new C0).
