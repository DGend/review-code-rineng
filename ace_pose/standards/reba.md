# REBA (Rapid Entire Body Assessment) — Technical Specification

**Version:** 1.0 — Clean-room re-implementation reference
**Standard:** Hignett & McAtamney 2000

---

## 1. Origin & References

The Rapid Entire Body Assessment (**REBA**) is a whole-body postural-load screening tool developed by **Sue Hignett** and **Lynn McAtamney** and published as a Technical Note in *Applied Ergonomics* **31**(2):201–205, 2000. It extends the upper-limb–focused RULA method to capture whole-body, unpredictable postures common in healthcare and service industries, and incorporates **load/force**, **coupling**, and **activity** modifiers.

REBA evolved from the **Quick Capture Method** (Hignett 1998) used for posture sampling in healthcare. The tool produces a single integer **Final REBA Score** (1–15) that maps to one of five **Action Levels** (0–4).

**Primary reference:**

- Hignett, S., McAtamney, L. (2000). "Rapid Entire Body Assessment (REBA)." *Applied Ergonomics* **31**(2): 201–205. DOI: `10.1016/S0003-6870(99)00039-3`. PMID: 10711982.

**Authoritative reproductions used to cross-check tables and thresholds:**

- *ErgoPlus* — Step-by-Step Guide and worksheet by Ergonomics Plus Inc. (based on the Cornell University worksheet by Dr. Alan Hedge).
- *Ergonautas* — Universidad Politécnica de Valencia, Diego-Mas, J. A., REBA help page.
- Cornell University Ergonomics Web (Hedge, A.) — Cornell REBA slides, Sept 2001.
- Physiopedia — REBA reference page.
- Institute of Ergonomics and Hygiene (IEH, Singapore) — REBA tool description.

---

## 2. Two Body-Segment Groups

REBA partitions the body into **two scoring groups** that are computed independently and combined via Table C.

- **Group A — Neck, Trunk, Legs.**
  Receives the **Load/Force** modifier. Group A's table output ∈ [1, 9]; after adding load/force (0–3) the resulting **Score A** ∈ [1, 12].

- **Group B — Upper Arm, Lower Arm, Wrist.**
  Receives the **Coupling** modifier. Group B's table output ∈ [1, 9]; after adding coupling (0–3) the resulting **Score B** ∈ [1, 12].

Group B is assessed **per side** (left/right); the higher (worse) side typically drives the final score.

Final REBA = Table C[Score A, Score B] + Activity Score, clamped to [1, 15].

---

## 3. Per-Segment Scoring Rules

All angles are in degrees. Sign convention: flexion is positive (forward for trunk/neck/arms), extension is negative.

### 3.1 Trunk (base score 1–5)

| Posture (sagittal flexion θ) | Score |
|---|---|
| Upright (θ = 0°) | 1 |
| 0° < |θ| ≤ 20° (flexion or extension) | 2 |
| 20° < θ ≤ 60° flexion, **or** any extension > 20° | 3 |
| θ > 60° flexion | 4 |

**Adjustment:** +1 if trunk is **twisted** OR **side-bent** (lateral flexion). Max trunk = 5.

### 3.2 Neck (base score 1–2)

| Posture | Score |
|---|---|
| 0° ≤ flexion ≤ 20° | 1 |
| Flexion > 20° **or** any extension | 2 |

**Adjustment:** +1 if **twisted** OR **side-bent**. Max neck = 3.

### 3.3 Legs (base score 1–2)

| Posture | Score |
|---|---|
| Bilateral weight-bearing, walking, or sitting | 1 |
| Unilateral weight-bearing, feather-weight bearing, or unstable | 2 |

**Knee-flexion adjustment** (NOT applied when sitting):

| Knee flexion | Adjustment |
|---|---|
| < 30° | +0 |
| 30°–60° | +1 |
| > 60° | +2 |

Max legs = 4.

### 3.4 Upper Arm (base score 1–4)

Shoulder–elbow flexion θ (positive = forward).

| Posture | Score |
|---|---|
| −20° ≤ θ ≤ 20° | 1 |
| θ > 20° extension, **or** 20° < θ ≤ 45° flexion | 2 |
| 45° < θ ≤ 90° flexion | 3 |
| θ > 90° flexion | 4 |

**Adjustments:**
- +1 if **shoulder is raised** (elevated)
- +1 if **upper arm is abducted** (or externally rotated)
- −1 if **leaning / arm supported / gravity-assisted**

Max upper arm = 6. (Floor at 1 is not formally specified, but implementations clamp ≥ 1.)

### 3.5 Lower Arm (base score 1–2)

Elbow flexion θ.

| Posture | Score |
|---|---|
| 60° ≤ θ ≤ 100° | 1 |
| θ < 60° **or** θ > 100° | 2 |

No adjustment. Max = 2.

### 3.6 Wrist (base score 1–2)

Wrist flexion/extension θ.

| Posture | Score |
|---|---|
| −15° ≤ θ ≤ 15° | 1 |
| θ < −15° **or** θ > 15° | 2 |

**Adjustment:** +1 if wrist is **bent from midline** (radial/ulnar deviation) **or twisted** (pronation/supination beyond neutral). Max wrist = 3.

---

## 4. Table A — Group A Subscore

Indexed by `Table_A[Trunk−1][Neck−1][Legs−1]`, where Trunk ∈ {1..5}, Neck ∈ {1..3}, Legs ∈ {1..4}. Output ∈ [1, 9].

| Trunk \ Neck | Neck=1, Legs 1 2 3 4 | Neck=2, Legs 1 2 3 4 | Neck=3, Legs 1 2 3 4 |
|:---:|:---:|:---:|:---:|
| **1** | 1, 2, 3, 4 | 1, 2, 3, 4 | 3, 3, 5, 6 |
| **2** | 2, 3, 4, 5 | 3, 4, 5, 6 | 4, 5, 6, 7 |
| **3** | 2, 4, 5, 6 | 4, 5, 6, 7 | 5, 6, 7, 8 |
| **4** | 3, 5, 6, 7 | 5, 6, 7, 8 | 6, 7, 8, 9 |
| **5** | 4, 6, 7, 8 | 6, 7, 8, 9 | 7, 8, 9, 9 |

Python-ready 3-D array (shape [Neck=3][Trunk=5][Legs=4]):

```python
TABLE_A = [
    # Neck = 1
    [[1,2,3,4], [2,3,4,5], [2,4,5,6], [3,5,6,7], [4,6,7,8]],
    # Neck = 2
    [[1,2,3,4], [3,4,5,6], [4,5,6,7], [5,6,7,8], [6,7,8,9]],
    # Neck = 3
    [[3,3,5,6], [4,5,6,7], [5,6,7,8], [6,7,8,9], [7,8,9,9]],
]
# Lookup: TABLE_A[neck-1][trunk-1][legs-1]
```

---

## 5. Load / Force Score (added to Table A output → Score A)

| Load | Score |
|---|---|
| < 5 kg (≈ < 11 lb) | 0 |
| 5–10 kg (≈ 11–22 lb) | 1 |
| > 10 kg (≈ > 22 lb) | 2 |

**Adjustment:** +1 if **shock** or **rapid build-up** of force.

`Score A = Table_A[neck][trunk][legs] + load_score`, clamped to [1, 12].

---

## 6. Table B — Group B Subscore

Indexed by `Table_B[UpperArm−1][LowerArm−1][Wrist−1]`. UpperArm ∈ {1..6}, LowerArm ∈ {1..2}, Wrist ∈ {1..3}. Output ∈ [1, 9].

| UpperArm \ LowerArm | LowerArm=1, Wrist 1 2 3 | LowerArm=2, Wrist 1 2 3 |
|:---:|:---:|:---:|
| **1** | 1, 2, 2 | 1, 2, 3 |
| **2** | 1, 2, 3 | 2, 3, 4 |
| **3** | 3, 4, 5 | 4, 5, 5 |
| **4** | 4, 5, 5 | 5, 6, 7 |
| **5** | 6, 7, 8 | 7, 8, 8 |
| **6** | 7, 8, 8 | 8, 9, 9 |

```python
TABLE_B = [
    [[1,2,2], [1,2,3]],  # UpperArm = 1
    [[1,2,3], [2,3,4]],  # UpperArm = 2
    [[3,4,5], [4,5,5]],  # UpperArm = 3
    [[4,5,5], [5,6,7]],  # UpperArm = 4
    [[6,7,8], [7,8,8]],  # UpperArm = 5
    [[7,8,8], [8,9,9]],  # UpperArm = 6
]
# Lookup: TABLE_B[upper-1][lower-1][wrist-1]
```

---

## 7. Coupling Score (added to Table B output → Score B)

| Quality | Score | Description |
|---|---|---|
| Good | 0 | Well-fitting handle and mid-range power grip |
| Fair | 1 | Hand-hold acceptable but not ideal, OR coupling acceptable via another body part |
| Poor | 2 | Hand-hold not acceptable, although possible |
| Unacceptable | 3 | No handles; awkward, unsafe; unsafe with any body part |

`Score B = Table_B[upper][lower][wrist] + coupling_score`, clamped to [1, 12].

---

## 8. Table C — Final Combined Subscore

Indexed by `Table_C[ScoreA−1][ScoreB−1]`. Both axes ∈ {1..12}. Output ∈ [1, 12].

| A\B | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|:---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **1** | 1 | 1 | 1 | 2 | 3 | 3 | 4 | 5 | 6 | 7 | 7 | 7 |
| **2** | 1 | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 6 | 7 | 7 | 8 |
| **3** | 2 | 3 | 3 | 3 | 4 | 5 | 6 | 7 | 7 | 8 | 8 | 8 |
| **4** | 3 | 4 | 4 | 4 | 5 | 6 | 7 | 8 | 8 | 9 | 9 | 9 |
| **5** | 4 | 4 | 4 | 5 | 6 | 7 | 8 | 8 | 9 | 9 | 9 | 9 |
| **6** | 6 | 6 | 6 | 7 | 8 | 8 | 9 | 9 | 10 | 10 | 10 | 10 |
| **7** | 7 | 7 | 7 | 8 | 9 | 9 | 9 | 10 | 10 | 11 | 11 | 11 |
| **8** | 8 | 8 | 8 | 9 | 10 | 10 | 10 | 10 | 10 | 11 | 11 | 11 |
| **9** | 9 | 9 | 9 | 10 | 10 | 10 | 11 | 11 | 11 | 12 | 12 | 12 |
| **10** | 10 | 10 | 10 | 11 | 11 | 11 | 11 | 12 | 12 | 12 | 12 | 12 |
| **11** | 11 | 11 | 11 | 11 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| **12** | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |

```python
TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]
# Lookup: TABLE_C[score_a-1][score_b-1]
```

---

## 9. Activity Score (added to Table C → Final REBA)

Add **+1 for each** condition that applies:

- **+1** if one or more body parts are **static** — held in the same position for **> 1 minute**.
- **+1** if action is **repetitive** — repeated small-range actions **more than 4× per minute** (walking excluded).
- **+1** if action causes **rapid, large-range changes in posture** or an **unstable base**.

Maximum activity score = 3. The three conditions are independent and may stack.

`Final REBA = Table_C[Score A][Score B] + Activity Score`, clamped to [1, 15].

---

## 10. Action Levels

| Action Level | REBA Score | Risk Level | Action |
|:---:|:---:|---|---|
| 0 | 1 | Negligible | None necessary |
| 1 | 2–3 | Low | May be necessary |
| 2 | 4–7 | Medium | Necessary |
| 3 | 8–10 | High | Necessary soon |
| 4 | 11–15 | Very High | Necessary **NOW** |

---

## 11. Computation from 3-D Joint Angles

Inputs are 3-D keypoints (e.g., from monocular or multi-view pose estimation). Define a **world frame** with `+Z` up, `+X` forward (worker-facing direction), `+Y` to the left. All angles use `atan2` in degrees.

For each segment, define a **proximal–distal vector** and compute its projection onto two anatomical planes:

- **Sagittal plane** = XZ plane (flexion/extension)
- **Coronal plane** = YZ plane (abduction/adduction, lateral bending)
- **Transverse plane** = XY plane (rotation/twist)

### 11.1 Trunk flexion
`v_trunk = midShoulder − midHip` (or C7 − sacrum).
`trunk_flex = atan2(v_trunk.x, v_trunk.z)` — signed; positive forward.
- `|v| ≤ 5°` → upright (score 1); else compare against {20°, 60°} thresholds.
- **Twist:** angle between shoulder-line vector (R_shoulder − L_shoulder) projected on XY and hip-line vector projected on XY. If > 10° → twisted.
- **Side-bend:** lateral component `atan2(v_trunk.y, v_trunk.z)`; > 10° → side-bent.

### 11.2 Neck flexion
`v_neck = head − midShoulder` (or nose − C7).
`neck_flex = atan2(v_neck.x, v_neck.z)`.
- Twist: head-yaw vs shoulder-yaw difference > 10°.
- Side-bend: lateral component > 10°.

### 11.3 Legs
- **Bilateral vs unilateral:** compare vertical projection of each foot to hip center; if both feet support load (z-position within tolerance) and lateral COM offset is small → bilateral (1); else unilateral (2).
- **Knee flexion:** angle between `v_thigh = knee − hip` and `v_shank = ankle − knee`; knee flexion = 180° − interior angle. Use the **more-flexed** knee.

### 11.4 Upper arm
`v_upper = elbow − shoulder`. Reference: trunk's distal axis (downward along trunk).
`upper_flex = signed angle between v_upper and (−trunk axis)` in the sagittal plane of the trunk.
- **Abduction:** out-of-sagittal-plane component of v_upper > 30° (some implementations use 45°; 30° is closer to the original adjustment cue).
- **Shoulder raised:** vertical displacement of the shoulder marker vs neutral shoulder height > ~5 cm, OR scapular elevation angle estimable from clavicle.
- **Supported/leaning:** external knowledge (cannot be derived from pose alone) — flag manually or via context.

### 11.5 Lower arm
`v_lower = wrist − elbow`.
`lower_flex = 180° − angle(v_upper, v_lower)` (i.e., elbow flexion).

### 11.6 Wrist
With **forearm + hand keypoints** present:
`v_hand = midKnuckle − wrist`, `wrist_flex = signed angle between v_hand and v_lower` in the hand's sagittal plane.
**Deviation:** lateral angle in the hand's coronal plane.
**Twist:** pronation/supination — requires two hand-plane markers (e.g., index-MCP and pinky-MCP); twist > ~45° from neutral palm-down/up.

### 11.7 Load, Coupling, Activity
These cannot be derived from pose; require task metadata, force sensors, video tag-in, or manual entry. Implementations should:
- accept `load_kg` numeric → map to 0/1/2 + shock flag.
- accept `coupling` enum → 0/1/2/3.
- compute **static** (window > 60 s with low joint-angle variance), **repetitive** (count zero-crossings of a key joint > 4/min), and **rapid/unstable** (max angular velocity threshold or COM acceleration) from a time series.

---

## 12. Wrist Scoring Without Finger Keypoints

Many production pose-estimation pipelines (BlazePose 33-pt, COCO-17, OpenPose body-25) lack finger/MCP keypoints. The wrist subscore must still be produced:

**Recommended fallback (degraded but conservative):**

1. **Wrist flexion/extension cannot be reliably estimated** without a hand vector. Two options:
   - **Option A — Conservative default:** Set base wrist score = 1 (assume neutral). Reasonable when the task does not involve obvious wrist deviation. Mark wrist score as low-confidence.
   - **Option B — Forearm twist proxy:** Estimate the wrist's orientation from the forearm twist (pronation/supination) using consecutive frames or shoulder-elbow-wrist 3-D geometry. If forearm shows extreme pronation/supination (> 60° from neutral) or wrist linear velocity exceeds a threshold during repetitive tasks, escalate to score 2.

2. **Deviation/twist adjustment:** Without finger keypoints, the +1 deviation adjustment is **not derivable**. Either (a) leave at 0 and document, or (b) accept an external boolean from task metadata (manual tagging, tool/handle type).

3. **Hand keypoint augmentation (preferred when available):** Run a hand-keypoint model (e.g., MediaPipe Hands) on the wrist ROI; with index-MCP and pinky-MCP keypoints, compute hand plane and apply full §11.6 formulas.

4. **Always log degraded confidence** on the wrist subscore in the output so downstream consumers can weight it appropriately.

---

## 13. Common Implementation Errors / Pitfalls

1. **Forgetting the activity score.** The most common omission. Implementations frequently report `Score C` as the final REBA, under-reporting risk by 1–3 points. The activity score is **not optional** and is what differentiates REBA from a pure posture index.
2. **Defaulting coupling to 0.** Many tools silently set coupling = 0 (Good) when not specified. This under-reports Score B by up to 3 points. Either require explicit input or default to **1 (Fair)** as a conservative middle ground and surface it as a configurable parameter.
3. **Skipping the side-bend / twist adjustments.** These +1's on trunk and neck are easy to miss when computing from pose; they require cross-plane analysis, not just sagittal flexion.
4. **Wrong Table-C indexing.** Table C is `[Score A][Score B]` — NOT `[B][A]`. The matrix is **not symmetric**: A=1,B=6 → 3, but A=6,B=1 → 6.
5. **Treating Group B as bilateral once.** Group B should be scored **separately for left and right**; the higher side drives the final score (or both should be reported).
6. **Trunk score collapsing extension into flexion.** Trunk extension > 20° goes to score 3, not the same bucket as flexion > 60°. Many implementations use `abs(angle)` and lose this distinction.
7. **Applying knee adjustment while sitting.** The +2 knee-flexion-over-60° adjustment is explicitly **not for sitting** per the original specification.
8. **Off-by-one indexing.** Tables are documented 1-indexed but stored 0-indexed in code. Always subtract 1 at lookup time and bounds-check.
9. **Floor/ceiling clamping.** Upper-arm score with leaning adjustment (−1) can produce 0; clamp to ≥ 1. Score A and Score B after modifiers should be clamped to [1, 12].
10. **Load thresholds in pounds vs kilograms.** Original paper uses kg; many US worksheets use lb (< 11 lb, 11–22 lb, > 22 lb). The ~5 / ~10 kg breakpoints are canonical.
11. **Final REBA clamp.** Cap at 15 (the published maximum); Table C maxes at 12 + 3 activity = 15.
12. **Static-detection window.** "> 1 minute" must use a sliding 60-second window with low joint-variance criteria, not a fixed analysis bin — otherwise a 59-second hold straddling a bin boundary is missed.

---

## 14. Authoritative URLs / DOIs

1. **Original paper (DOI):** Hignett, S., McAtamney, L. (2000) "Rapid Entire Body Assessment (REBA)". *Applied Ergonomics* 31(2):201–205. https://doi.org/10.1016/S0003-6870(99)00039-3
2. **PubMed entry:** https://pubmed.ncbi.nlm.nih.gov/10711982/
3. **Cornell University, Prof. Alan Hedge — REBA slides (Sept 2001):** https://ergo.human.cornell.edu/Pub/AHquest/Cornell_REBA.pdf
4. **Cornell Ergonomics — REBA worksheet (Excel):** https://ergo.human.cornell.edu/CUErgoTools/REBA%206.xls
5. **ErgoPlus — REBA Step-by-Step Guide (PDF):** http://ergo-plus.com/wp-content/uploads/REBA-A-Step-by-Step-Guide.pdf
6. **ErgoPlus — REBA Employee Assessment Worksheet (PDF):** https://ergo-plus.com/wp-content/uploads/REBA.pdf
7. **Ergonautas (UPV) — REBA help page:** https://www.ergonautas.upv.es/metodos/reba/reba-ayuda.php
8. **Ergonautas — Online REBA calculator:** https://www.ergonautas.upv.es/ergoniza/app_en/land/index.html?method=reba
9. **Physiopedia — REBA reference:** https://www.physio-pedia.com/Rapid_Entire_Body_Assessment_(REBA)
10. **Institute of Ergonomics and Hygiene (IEH) — REBA tool:** https://ieh.sg/tool/reba/
11. **Open-source reference implementation (rs9000/ergonomics):** https://github.com/rs9000/ergonomics
12. **Literature review — Madani & Dababneh (2016) "REBA: A Literature Review":** https://thescipub.com/pdf/ajeassp.2016.107.118.pdf
13. **Hignett, S. (1998) "Quick Capture Method" — Posture sampling in patient handling:** *Applied Ergonomics* 29(4):285–290. DOI: `10.1016/S0003-6870(97)00088-1`

---

**End of Specification.**
