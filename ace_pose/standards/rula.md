# RULA (Rapid Upper Limb Assessment) — Technical Specification

> Clean-room re-implementation reference. Compiled from the McAtamney & Corlett 1993 paper, the Cornell/Hedge worksheet (2000/2001), the ErgoPlus step-by-step guide, the Humanics-ES reproduction, and Ergonautas (Universidad Politécnica de Valencia). All table values were cross-checked against at least two of these sources.

---

## 1. Origin & References

RULA was developed by Lynn McAtamney and E. Nigel Corlett at the Institute for Occupational Ergonomics, University of Nottingham, and first published as:

- **McAtamney, L. & Corlett, E. N. (1993).** *RULA: a survey method for the investigation of work-related upper limb disorders.* Applied Ergonomics, 24(2), 91–99. DOI: 10.1016/0003-6870(93)90080-S.
- **McAtamney, L. & Corlett, E. N. (2004).** *Rapid Upper Limb Assessment (RULA).* In N. Stanton et al. (Eds.), *Handbook of Human Factors and Ergonomics Methods*, Chapter 7. Boca Raton, FL: CRC Press. DOI: 10.1201/9780203489925.ch7.
- **Corlett, E. N. & Bishop, R. P. (1976).** *A technique for assessing postural discomfort.* Ergonomics, 19(2), 175–182. DOI: 10.1080/00140137608931530.

Authoritative reproductions used to cross-check tables:
- Cornell University Ergonomics Web (CUErgo / A. Hedge): https://ergo.human.cornell.edu/ahRULA.html — official PDF worksheet at https://ergo.human.cornell.edu/Pub/AHquest/RULAworksheet.pdf
- ErgoPlus worksheet: https://ergo-plus.com/wp-content/uploads/RULA.pdf
- Ergonautas (UPV): https://www.ergonautas.upv.es
- Humanics-ES reproduction (Table A): https://www.humanics-es.com/rula-9.htm
- Original 1993 paper (open mirror): http://ftp.demec.ufpr.br/disciplinas/TM802/RULA_original%201993.pdf
- PubMed record: https://pubmed.ncbi.nlm.nih.gov/15676903/

---

## 2. Two Body-Segment Groups

**Group A — Arms and Wrist** (scored per arm):
1. Upper arm (shoulder flexion/extension + adjustments)
2. Lower arm (elbow flexion + adjustments)
3. Wrist (flexion/extension + deviation)
4. Wrist twist (forearm pronation/supination)

**Group B — Neck, Trunk, Legs**:
5. Neck (cervical flexion/extension + twist/lateral)
6. Trunk (thoraco-lumbar flexion/extension + twist/lateral)
7. Legs (binary supported / unsupported)

Group A → Table A → Posture A → +muscle +force → **Wrist & Arm Score**.
Group B → Table B → Posture B → +muscle +force → **Neck/Trunk/Leg Score**.
Both feed Table C for the final grand score.

---

## 3. Per-Segment Scoring Rules

All angles in degrees. Flexion = positive (forward of trunk for arm, forward of vertical for neck/trunk).

### 3.1 Upper arm (Step 1)
| Position | Score |
|---|---|
| -20° ≤ θ ≤ 20° | **1** |
| > 20° extension, OR 20°–45° flexion | **2** |
| 45°–90° flexion | **3** |
| > 90° flexion | **4** |

Adjustments: +1 shoulder raised; +1 abducted; −1 supported/leaning. Clamp to [1, 6].

### 3.2 Lower arm (Step 2)
| Elbow flexion | Score |
|---|---|
| 60°–100° | **1** |
| < 60° OR > 100° | **2** |

Adjustments: +1 if crosses midline; +1 if out to side. Clamp to [1, 3].

### 3.3 Wrist (Step 3)
| |θ| (flex or ext) | Score |
|---|---|
| 0° (neutral) | **1** |
| 0°–15° | **2** |
| > 15° | **3** |

Adjustment: +1 if radial/ulnar deviation from midline. Clamp to [1, 4].

### 3.4 Wrist twist (Step 4)
- Mid-range: **1**
- Near end of range: **2**

### 3.5 Neck (Step 9)
| Position | Score |
|---|---|
| 0°–10° flexion | **1** |
| 10°–20° flexion | **2** |
| > 20° flexion | **3** |
| Any extension | **4** |

Adjustments: +1 twist; +1 side-bending. Clamp to [1, 6].

### 3.6 Trunk (Step 10)
| Position | Score |
|---|---|
| 0° upright | **1** |
| 0°–20° flexion | **2** |
| 20°–60° flexion | **3** |
| > 60° flexion | **4** |

Adjustments: +1 twist; +1 side-bending. Clamp to [1, 6].

### 3.7 Legs (Step 11)
- Supported & balanced: **1**
- Not supported / unbalanced: **2**

---

## 4. Table A — Arm & Wrist Subscore (6 × 3 × 4 × 2 = 144 entries)

**Correction (2026-10-04).** The version of this table first transcribed here on 2026-05-14 had 31 cells one point too low
(30 cells by 1, 1 cell by 2), 29 of them in Upper Arm rows 2-4. `ergo_v2/tables.py` was corrected on
2026-08-11 but this file was not. The values below are the corrected table. They agree in all 144 cells with the
independent legacy ETRI implementation and with the project ergonomist's hand lookups (22 of 22, including all 9 rows
where the old and corrected tables differ). `ace_pose/tests/test_ergo_v2_golden.py::test_rula_table_a_cells` now pins
every cell, because the earlier tests checked only the table's shape.

`TableA[upper_arm][lower_arm][wrist][wrist_twist]`.

```
Upper Arm = 1
  LA=1:  1, 2, 2, 2, 2, 3, 3, 3
  LA=2:  2, 2, 2, 2, 3, 3, 3, 3
  LA=3:  2, 3, 3, 3, 3, 3, 4, 4

Upper Arm = 2
  LA=1:  2, 3, 3, 3, 3, 4, 4, 4
  LA=2:  3, 3, 3, 3, 3, 4, 4, 4
  LA=3:  3, 4, 4, 4, 4, 4, 5, 5

Upper Arm = 3
  LA=1:  3, 3, 4, 4, 4, 4, 5, 5
  LA=2:  3, 4, 4, 4, 4, 4, 5, 5
  LA=3:  4, 4, 4, 4, 4, 5, 5, 5

Upper Arm = 4
  LA=1:  4, 4, 4, 4, 4, 5, 5, 5
  LA=2:  4, 4, 4, 4, 4, 5, 5, 5
  LA=3:  4, 4, 4, 5, 5, 5, 6, 6

Upper Arm = 5
  LA=1:  5, 5, 5, 5, 5, 6, 6, 7
  LA=2:  5, 6, 6, 6, 6, 7, 7, 7
  LA=3:  6, 6, 6, 7, 7, 7, 7, 8

Upper Arm = 6
  LA=1:  7, 7, 7, 7, 7, 8, 8, 9
  LA=2:  8, 8, 8, 8, 8, 9, 9, 9
  LA=3:  9, 9, 9, 9, 9, 9, 9, 9
```

Each row's 8 values are laid out as `(W1 T1, W1 T2, W2 T1, W2 T2, W3 T1, W3 T2, W4 T1, W4 T2)`.

```python
TABLE_A = {  # (upper, lower, wrist, twist) -> posture A
  (1,1,1,1):1,(1,1,1,2):2,(1,1,2,1):2,(1,1,2,2):2,(1,1,3,1):2,(1,1,3,2):3,(1,1,4,1):3,(1,1,4,2):3,
  (1,2,1,1):2,(1,2,1,2):2,(1,2,2,1):2,(1,2,2,2):2,(1,2,3,1):3,(1,2,3,2):3,(1,2,4,1):3,(1,2,4,2):3,
  (1,3,1,1):2,(1,3,1,2):3,(1,3,2,1):3,(1,3,2,2):3,(1,3,3,1):3,(1,3,3,2):3,(1,3,4,1):4,(1,3,4,2):4,
  (2,1,1,1):2,(2,1,1,2):3,(2,1,2,1):3,(2,1,2,2):3,(2,1,3,1):3,(2,1,3,2):4,(2,1,4,1):4,(2,1,4,2):4,
  (2,2,1,1):3,(2,2,1,2):3,(2,2,2,1):3,(2,2,2,2):3,(2,2,3,1):3,(2,2,3,2):4,(2,2,4,1):4,(2,2,4,2):4,
  (2,3,1,1):3,(2,3,1,2):4,(2,3,2,1):4,(2,3,2,2):4,(2,3,3,1):4,(2,3,3,2):4,(2,3,4,1):5,(2,3,4,2):5,
  (3,1,1,1):3,(3,1,1,2):3,(3,1,2,1):4,(3,1,2,2):4,(3,1,3,1):4,(3,1,3,2):4,(3,1,4,1):5,(3,1,4,2):5,
  (3,2,1,1):3,(3,2,1,2):4,(3,2,2,1):4,(3,2,2,2):4,(3,2,3,1):4,(3,2,3,2):4,(3,2,4,1):5,(3,2,4,2):5,
  (3,3,1,1):4,(3,3,1,2):4,(3,3,2,1):4,(3,3,2,2):4,(3,3,3,1):4,(3,3,3,2):5,(3,3,4,1):5,(3,3,4,2):5,
  (4,1,1,1):4,(4,1,1,2):4,(4,1,2,1):4,(4,1,2,2):4,(4,1,3,1):4,(4,1,3,2):5,(4,1,4,1):5,(4,1,4,2):5,
  (4,2,1,1):4,(4,2,1,2):4,(4,2,2,1):4,(4,2,2,2):4,(4,2,3,1):4,(4,2,3,2):5,(4,2,4,1):5,(4,2,4,2):5,
  (4,3,1,1):4,(4,3,1,2):4,(4,3,2,1):4,(4,3,2,2):5,(4,3,3,1):5,(4,3,3,2):5,(4,3,4,1):6,(4,3,4,2):6,
  (5,1,1,1):5,(5,1,1,2):5,(5,1,2,1):5,(5,1,2,2):5,(5,1,3,1):5,(5,1,3,2):6,(5,1,4,1):6,(5,1,4,2):7,
  (5,2,1,1):5,(5,2,1,2):6,(5,2,2,1):6,(5,2,2,2):6,(5,2,3,1):6,(5,2,3,2):7,(5,2,4,1):7,(5,2,4,2):7,
  (5,3,1,1):6,(5,3,1,2):6,(5,3,2,1):6,(5,3,2,2):7,(5,3,3,1):7,(5,3,3,2):7,(5,3,4,1):7,(5,3,4,2):8,
  (6,1,1,1):7,(6,1,1,2):7,(6,1,2,1):7,(6,1,2,2):7,(6,1,3,1):7,(6,1,3,2):8,(6,1,4,1):8,(6,1,4,2):9,
  (6,2,1,1):8,(6,2,1,2):8,(6,2,2,1):8,(6,2,2,2):8,(6,2,3,1):8,(6,2,3,2):9,(6,2,4,1):9,(6,2,4,2):9,
  (6,3,1,1):9,(6,3,1,2):9,(6,3,2,1):9,(6,3,2,2):9,(6,3,3,1):9,(6,3,3,2):9,(6,3,4,1):9,(6,3,4,2):9,
}
```

---

## 5. Table B — Neck, Trunk & Legs Subscore (6 × 6 × 2 = 72 entries)

```
                Trunk=1   Trunk=2   Trunk=3   Trunk=4   Trunk=5   Trunk=6
                L1  L2    L1  L2    L1  L2    L1  L2    L1  L2    L1  L2
Neck=1:          1   3     2   3     3   4     5   5     6   6     7   7
Neck=2:          2   3     2   3     4   5     5   5     6   7     7   7
Neck=3:          3   3     3   4     4   5     5   6     6   7     7   7
Neck=4:          5   5     5   6     6   7     7   7     7   7     8   8
Neck=5:          7   7     7   7     7   8     8   8     8   8     8   8
Neck=6:          8   8     8   8     8   8     8   9     9   9     9   9
```

```python
TABLE_B = {
  (1,1,1):1,(1,1,2):3,(1,2,1):2,(1,2,2):3,(1,3,1):3,(1,3,2):4,(1,4,1):5,(1,4,2):5,(1,5,1):6,(1,5,2):6,(1,6,1):7,(1,6,2):7,
  (2,1,1):2,(2,1,2):3,(2,2,1):2,(2,2,2):3,(2,3,1):4,(2,3,2):5,(2,4,1):5,(2,4,2):5,(2,5,1):6,(2,5,2):7,(2,6,1):7,(2,6,2):7,
  (3,1,1):3,(3,1,2):3,(3,2,1):3,(3,2,2):4,(3,3,1):4,(3,3,2):5,(3,4,1):5,(3,4,2):6,(3,5,1):6,(3,5,2):7,(3,6,1):7,(3,6,2):7,
  (4,1,1):5,(4,1,2):5,(4,2,1):5,(4,2,2):6,(4,3,1):6,(4,3,2):7,(4,4,1):7,(4,4,2):7,(4,5,1):7,(4,5,2):7,(4,6,1):8,(4,6,2):8,
  (5,1,1):7,(5,1,2):7,(5,2,1):7,(5,2,2):7,(5,3,1):7,(5,3,2):8,(5,4,1):8,(5,4,2):8,(5,5,1):8,(5,5,2):8,(5,6,1):8,(5,6,2):8,
  (6,1,1):8,(6,1,2):8,(6,2,1):8,(6,2,2):8,(6,3,1):8,(6,3,2):8,(6,4,1):8,(6,4,2):9,(6,5,1):9,(6,5,2):9,(6,6,1):9,(6,6,2):9,
}
```

---

## 6. Muscle Use and Force/Load Adjustments

### 6.1 Muscle use score (Group A and B, independently)
- **+1** if static (>1 min) OR repetitive (≥4×/min). Otherwise 0.
- Not additive — static AND repetitive together is still +1.

### 6.2 Force/Load score
| Load condition | Score |
|---|---|
| < 2 kg, intermittent | **0** |
| 2–10 kg, intermittent | **1** |
| 2–10 kg, static/repeated, OR > 10 kg intermittent | **2** |
| > 10 kg static/repeated, OR shock/sudden force | **3** |

### 6.3 Combined scores
- Wrist & Arm score = Posture A + Muscle A + Force A
- Neck/Trunk/Leg score = Posture B + Muscle B + Force B

---

## 7. Table C — Final Grand Score (8 × 7 = 56 entries)

```
                  NTL=1  NTL=2  NTL=3  NTL=4  NTL=5  NTL=6  NTL=7+
Wrist/Arm = 1       1      2      3      3      4      5      5
Wrist/Arm = 2       2      2      3      4      4      5      5
Wrist/Arm = 3       3      3      3      4      4      5      6
Wrist/Arm = 4       3      3      3      4      5      6      6
Wrist/Arm = 5       4      4      4      5      6      7      7
Wrist/Arm = 6       4      4      5      6      6      7      7
Wrist/Arm = 7       5      5      6      6      7      7      7
Wrist/Arm = 8+      5      5      6      7      7      7      7
```

```python
TABLE_C = [
  [None,None,None,None,None,None,None,None],   # sentinel
  [None, 1, 2, 3, 3, 4, 5, 5],   # WA=1
  [None, 2, 2, 3, 4, 4, 5, 5],   # WA=2
  [None, 3, 3, 3, 4, 4, 5, 6],   # WA=3
  [None, 3, 3, 3, 4, 5, 6, 6],   # WA=4
  [None, 4, 4, 4, 5, 6, 7, 7],   # WA=5
  [None, 4, 4, 5, 6, 6, 7, 7],   # WA=6
  [None, 5, 5, 6, 6, 7, 7, 7],   # WA=7
  [None, 5, 5, 6, 7, 7, 7, 7],   # WA=8+
]
```

---

## 8. Action Levels

| Grand Score | Action Level | Interpretation |
|---|---|---|
| 1–2 | **1** | Acceptable if not maintained for long. |
| 3–4 | **2** | Further investigation; changes may be required. |
| 5–6 | **3** | Investigation and changes required **soon**. |
| 7+ | **4** | Investigation and changes required **immediately**. |

---

## 9. Computation From 3-D Joint Angles

Notation: `P_x` is a 3-D point. `v(A,B) = B − A`. `angle(u, v) = atan2(‖u×v‖, u·v)` ∈ [0°, 180°].

### 9.1 Body reference frame
- `pelvis = midpoint(left_hip, right_hip)`
- `neck = midpoint(left_shoulder, right_shoulder)`
- `up_body = normalize(neck − pelvis)`
- `side = normalize(right_shoulder − left_shoulder)`
- `fwd = normalize(cross(side, up_body))`

### 9.2 Upper-arm angle
- `v_trunk = pelvis − shoulder`
- `v_arm = elbow − shoulder`
- Project both to sagittal plane (subtract `(v·side)·side`).
- `θ_upper = signed_angle(v_trunk_proj, v_arm_proj)` around `side`, positive when arm anterior.
- Shoulder raised (+1) if shoulder Y above neck level (>2 cm).
- Abducted (+1) if frontal-plane angle from `−up_body` exceeds ~30°.
- Supported (−1) is contextual.

### 9.3 Lower-arm angle
- `v_eu = shoulder − elbow`, `v_ew = wrist − elbow`
- `θ_lower = angle(v_eu, v_ew)`, score by thresholds 60° and 100°.
- Cross midline (+1) if wrist lateral coord opposite-side of pelvis.
- Out to side (+1) if wrist lateral exceeds shoulder lateral by ~10 cm.

### 9.4 Wrist angle (requires hand keypoint)
- `v_we = elbow − wrist`, `v_wh = middle_MCP − wrist` (or hand center).
- `θ_wrist = 180° − angle(v_we, v_wh)`, neutral = 0°.

### 9.5 Wrist twist
Requires hand-orientation cue (knuckle plane). Default = 1 if unavailable.

### 9.6 Neck angle
- `v_head = head_top − neck`
- `θ_neck = signed_angle(up_world, v_head)` around `side`.
- Twist (+1) if shoulder-line yaw differs from head-line yaw by >10°.
- Side bend (+1) if `v_head` frontal projection deviates from vertical by ≥10°.

### 9.7 Trunk angle and twist
- `v_trunk_up = neck − pelvis`
- `θ_trunk = signed_angle(up_world, v_trunk_up)`.
- Twist: angle between shoulder-line and hip-line projected to transverse plane.

### 9.8 Legs
Approximate from ankle vs. pelvis offset, knee angles, and inter-knee match.

---

## 10. Wrist Scoring Without Finger Keypoints

When using a body-only pose model (no hand/MCP keypoints):

1. **Wrist score = 1** (neutral). Until 2026-10-08 the default was 2, the conservative choice of Cornell CUErgo when wrist data are missing; the project ergonomist scores an unseen wrist as neutral (all 22 pilot images), so the default follows that practice.
2. **Wrist twist = 1** (mid-range default).
3. **Wrist deviation adjustment = 0**.
4. Expose as parameters: `wrist_score_default: int = 1`, `wrist_twist_default: int = 1`, `wrist_deviation_default: int = 0`.

If a separate hand-pose model is available, compute proper angles. Without it, do not pretend to know.

---

## 11. Common Implementation Errors and Pitfalls

1. **Wrong vector for upper-arm flexion.** Use `shoulder→elbow`, never `shoulder→wrist`.
2. **Confusing neck with trunk angle.** Neck = head vs. vertical; Trunk = up_body vs. vertical. Using head vs. up_body collapses neck when trunk is bent.
3. **Sign convention for extension.** Upper-arm score 2 fires for both 20°–45° flexion AND > 20° extension; many forget the extension branch.
4. **Off-by-one in Lower Arm bounds.** Treat the **open** interval (60°, 100°) as score 1 to match Cornell/ErgoPlus.
5. **Trunk = 1 only when 0°.** Many implementations score 1 for [0°, 20°] and skip score 2.
6. **Twist adjustments confused between groups.** Trunk twist → trunk score (+1), Neck twist → neck score (+1). Each segment gets its own.
7. **Applying muscle/force to grand score.** Muscle and force adjust Posture A and Posture B *separately*. The grand score is read from Table C — DO NOT add muscle/force again.
8. **Table C clamping.** Wrist/Arm > 8 uses row 8+; Neck/Trunk/Leg > 7 uses column 7+.
9. **Side selection.** Score both sides; the worst arm drives the grand score.
10. **Deviation vs. twist confusion.** Deviation = radial/ulnar (+1 to wrist score). Twist = forearm pronation/supination (its own subscore 1/2).
11. **Upper-arm threshold version drift.** Use ±20° (1993 paper canonical), not ±15° (old Cornell 2000 printing).
12. **Boundary inclusivity.** At exactly the boundary value, score the lower zone. Document explicitly.

---

## 12. Authoritative URLs / DOIs

1. McAtamney & Corlett (1993), Applied Ergonomics 24(2):91–99 — DOI: https://doi.org/10.1016/0003-6870(93)90080-S — Mirror: http://ftp.demec.ufpr.br/disciplinas/TM802/RULA_original%201993.pdf
2. McAtamney & Corlett (2004), Handbook ch. 7 — DOI: https://doi.org/10.1201/9780203489925.ch7
3. Corlett & Bishop (1976), Ergonomics 19(2):175–182 — DOI: https://doi.org/10.1080/00140137608931530
4. Cornell CUErgo RULA: https://ergo.human.cornell.edu/ahRULA.html
5. Cornell RULA worksheet PDF: https://ergo.human.cornell.edu/Pub/AHquest/RULAworksheet.pdf
6. ErgoPlus RULA worksheet PDF: https://ergo-plus.com/wp-content/uploads/RULA.pdf
7. ErgoPlus step-by-step guide: https://ergo-plus.com/rula-assessment-tool-guide/
8. Ergonautas (UPV) RULA: https://www.ergonautas.upv.es
9. Humanics-ES Table A: https://www.humanics-es.com/rula-9.htm
10. PubMed: https://pubmed.ncbi.nlm.nih.gov/15676903/
11. ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/000368709390080S
