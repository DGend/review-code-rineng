# OWAS (Ovako Working Posture Analysis System) — Technical Specification

A clean-room implementation reference. Version targeted at deriving OWAS scores from 3D
human pose data (e.g., MediaPipe, OpenPose, BlazePose, SMPL/SMPL-X keypoints).

---

## 1. Origin and References

OWAS was developed in Finland in the mid-1970s by Osmo Karhu, Pekka Kansi, and
Ilkka Kuorinka at the Finnish Institute of Occupational Health, working with the steel
producer Ovako Oy. The method was published as:

- **Karhu O., Kansi P., Kuorinka I. (1977).** "Correcting working postures in industry:
  A practical method for analysis." *Applied Ergonomics*, **8(4)**, 199–201.
  DOI: 10.1016/0003-6870(77)90164-8.

It was constructed by photographing 72 representative postures in the Ovako steel works,
which were then grouped into the canonical four-axis taxonomy (back × arms × legs × load).
Validation and extension followed in:

- **Karhu O., Härkönen R., Sorvali P., Vepsalainen P. (1981).** "Observing working postures
  in industry: Examples of OWAS application." *Applied Ergonomics*, **12(1)**, 13–17.
  DOI: 10.1016/0003-6870(81)90088-0.
- **Stoffert G. (1985).** "Analyse und Einstufung von Körperhaltungen bei der Arbeit nach
  der OWAS-Methode." *Zeitschrift für Arbeitswissenschaft*, **39**, 31–38. Introduced
  refinements to angular thresholds for back posture (especially the bent/twisted
  combination).
- **Louhevaara V., Suurnäkki T. (1992).** *OWAS: A method for the evaluation of postural
  load during work.* Helsinki: Institute of Occupational Health & Centre for Occupational
  Safety. The canonical training manual that fixes the 252-entry action-category lookup
  table still in use today.

Modern reference materials:
- **Diego-Mas J.A. (Ergonautas, Universidad Politécnica de Valencia).** "Método OWAS —
  Ovako Working Analysis System." The de facto online reference for the Spanish/English
  speaking ergonomics community; mirrors the Louhevaara & Suurnäkki tables.
- **ErgoPlus / ErgoIBV** training documentation (modules on OWAS).
- General textbooks: **Bridger R.S., *Introduction to Ergonomics* (3rd ed., CRC, 2008);
  Salvendy G., *Handbook of Human Factors and Ergonomics* (Wiley, 5th ed., 2021)**;
  both contain OWAS chapters.
- **Gómez-Galán M. et al. (2017)** "Musculoskeletal disorders: OWAS review."
  *International Journal of Industrial Ergonomics* — recent peer-reviewed survey
  (PMC5546841) covering validation across 53 studies.

OSHA does not publish OWAS as a regulatory standard; it is referenced indirectly through
NIOSH ergonomic-assessment guidance for musculoskeletal disorder prevention.

---

## 2. Coding Scheme — the 4-Digit Code `[Back, Arms, Legs, Load]`

Every observation is encoded as a 4-digit posture code. The first digit describes the
trunk (back), the second describes the arms, the third describes the legs, and the
fourth describes the handled load or applied force. Domains:
**Back ∈ {1,2,3,4}**, **Arms ∈ {1,2,3}**, **Legs ∈ {1,2,3,4,5,6,7}**,
**Load ∈ {1,2,3}** → **4 × 3 × 7 × 3 = 252 unique codes.**

### 2.1 Back (digit 1)

| Code | Posture | Definition (angle threshold) |
|------|---------|------------------------------|
| 1    | Straight | Trunk aligned with the hips-legs axis. Sagittal flexion ≤ 20°, and axial twist / lateral bend ≤ 20°. |
| 2    | Bent     | Trunk flexion (forward or backward) > 20° in the sagittal plane, no significant twist. |
| 3    | Twisted  | Axial rotation OR lateral bending > 20°, no significant sagittal flexion. |
| 4    | Bent AND twisted | Sagittal flexion > 20° **and** simultaneously axial rotation OR lateral bend > 20°. |

The 20° threshold is the canonical value from Karhu et al. (1977), confirmed by
Louhevaara & Suurnäkki (1992) and Ergonautas.

### 2.2 Arms (digit 2)

| Code | Posture | Definition |
|------|---------|------------|
| 1    | Both arms below shoulder | Both glenohumeral (shoulder) joints kept below the shoulder line. |
| 2    | One arm at or above shoulder | Exactly one shoulder elevated to or above the acromion (shoulder-line) horizontal plane. |
| 3    | Both arms at or above shoulder | Both shoulders elevated to or above the acromion line. |

"Shoulder line" = the horizontal plane through the acromion processes. The pragmatic
rule used in implementations is to compare the y-coordinate of the wrist/hand relative
to the shoulder.

### 2.3 Legs (digit 3)

| Code | Posture | Definition (angle threshold) |
|------|---------|------------------------------|
| 1    | Sitting | Trunk supported by buttocks on a seat. Hip flexion ≈ 90°, both knees flexed. |
| 2    | Standing, both legs straight | Both knees ≥ 150° (flexion ≤ 30°). Body weight ~ evenly distributed. |
| 3    | Standing, weight on one straight leg | One knee ≥ 150° bearing weight; the other slightly flexed or off the ground. Lateral CoM offset > 0.1 of body width. |
| 4    | Both knees bent (squatting/crouching), weight balanced | Both knees < 150°, weight on both feet. |
| 5    | Both knees bent, weight on one leg | Both knees < 150°, body weight predominantly on one leg. |
| 6    | Kneeling | One or both knees touch the ground. |
| 7    | Walking | Locomotion (forward CoM velocity, alternating step length). |

The 150° threshold is canonical from Karhu et al. (1977). It corresponds to about 30°
of knee flexion from full extension.

**Implementation note (2026-10-05).** Codes 3 and 5 depend on how body weight is split between the legs,
which a single monocular frame does not show. The earlier implementation inferred it from left-right ankle
asymmetry about the pelvis; that labelled 84% of the 6,007-pose cohort as one-legged and disagreed with the
project ergonomist on all 6 straight-legged standing poses it flagged among the 22 expert-scored images.
`legs_balanced` now assumes bilateral support (as the walking flag defaults to not walking). Code 3 is still
assigned when exactly one knee is bent, which is observable. RULA and REBA leg scores use the same function and
accept a caller-supplied `legs_supported` override.

### 2.4 Load / Force (digit 4)

| Code | Range |
|------|-------|
| 1    | Load or force < 10 kg (≈ < 98 N) |
| 2    | 10 kg ≤ Load ≤ 20 kg (≈ 98–196 N) |
| 3    | Load > 20 kg (≈ > 196 N) |

For applied force (push/pull), convert to an equivalent static load using
F (N) / 9.81 (m/s²) = mass-equivalent (kg).

---

## 3. Action Category Lookup Table (Canonical 252-Entry Matrix)

Indexed as `AC[back-1][arms-1][legs-1][load-1]`. AC values ∈ {1, 2, 3, 4}.
Cross-validated against Louhevaara & Suurnäkki (1992), Ergonautas, and the open-source
`rs9000/ergonomics` Python implementation.

### Back = 1 (Straight)
```
Arms=1: legs1=[1,1,1] legs2=[1,1,1] legs3=[1,1,1] legs4=[2,2,2] legs5=[2,2,2] legs6=[1,1,1] legs7=[1,1,1]
Arms=2: legs1=[1,1,1] legs2=[1,1,1] legs3=[1,1,1] legs4=[2,2,2] legs5=[2,2,2] legs6=[1,1,1] legs7=[1,1,1]
Arms=3: legs1=[1,1,1] legs2=[1,1,1] legs3=[1,1,1] legs4=[2,2,3] legs5=[2,2,3] legs6=[1,1,1] legs7=[1,1,2]
```

### Back = 2 (Bent)
```
Arms=1: legs1=[2,2,3] legs2=[2,2,3] legs3=[2,2,3] legs4=[3,3,3] legs5=[3,3,3] legs6=[2,2,2] legs7=[2,3,3]
Arms=2: legs1=[2,2,3] legs2=[2,2,3] legs3=[2,3,3] legs4=[3,4,4] legs5=[3,4,4] legs6=[3,3,4] legs7=[2,3,4]
Arms=3: legs1=[3,3,4] legs2=[2,2,3] legs3=[3,3,3] legs4=[3,4,4] legs5=[4,4,4] legs6=[4,4,4] legs7=[2,3,4]
```

### Back = 3 (Twisted)
```
Arms=1: legs1=[1,1,1] legs2=[1,1,1] legs3=[1,1,2] legs4=[3,3,3] legs5=[4,4,4] legs6=[1,1,1] legs7=[1,1,1]
Arms=2: legs1=[2,2,3] legs2=[1,1,1] legs3=[1,1,2] legs4=[4,4,4] legs5=[4,4,4] legs6=[3,3,3] legs7=[1,1,1]
Arms=3: legs1=[2,2,3] legs2=[1,1,1] legs3=[2,3,3] legs4=[4,4,4] legs5=[4,4,4] legs6=[4,4,4] legs7=[1,1,2]
```

### Back = 4 (Bent and Twisted)
```
Arms=1: legs1=[2,3,3] legs2=[2,2,3] legs3=[2,3,3] legs4=[4,4,4] legs5=[4,4,4] legs6=[4,4,4] legs7=[2,3,4]
Arms=2: legs1=[3,3,4] legs2=[2,3,4] legs3=[3,3,4] legs4=[4,4,4] legs5=[4,4,4] legs6=[4,4,4] legs7=[2,3,4]
Arms=3: legs1=[4,4,4] legs2=[2,3,4] legs3=[3,3,4] legs4=[4,4,4] legs5=[4,4,4] legs6=[4,4,4] legs7=[2,3,4]
```

Within each triple, the three numbers correspond to load = 1, 2, 3 respectively.

### Python initialisation (verbatim, recommended for implementation):

```python
import numpy as np
OWAS_AC = np.ones((4, 3, 7, 3), dtype=int)

OWAS_AC[0] = [
    [[1,1,1],[1,1,1],[1,1,1],[2,2,2],[2,2,2],[1,1,1],[1,1,1]],
    [[1,1,1],[1,1,1],[1,1,1],[2,2,2],[2,2,2],[1,1,1],[1,1,1]],
    [[1,1,1],[1,1,1],[1,1,1],[2,2,3],[2,2,3],[1,1,1],[1,1,2]],
]
OWAS_AC[1] = [
    [[2,2,3],[2,2,3],[2,2,3],[3,3,3],[3,3,3],[2,2,2],[2,3,3]],
    [[2,2,3],[2,2,3],[2,3,3],[3,4,4],[3,4,4],[3,3,4],[2,3,4]],
    [[3,3,4],[2,2,3],[3,3,3],[3,4,4],[4,4,4],[4,4,4],[2,3,4]],
]
OWAS_AC[2] = [
    [[1,1,1],[1,1,1],[1,1,2],[3,3,3],[4,4,4],[1,1,1],[1,1,1]],
    [[2,2,3],[1,1,1],[1,1,2],[4,4,4],[4,4,4],[3,3,3],[1,1,1]],
    [[2,2,3],[1,1,1],[2,3,3],[4,4,4],[4,4,4],[4,4,4],[1,1,2]],
]
OWAS_AC[3] = [
    [[2,3,3],[2,2,3],[2,3,3],[4,4,4],[4,4,4],[4,4,4],[2,3,4]],
    [[3,3,4],[2,3,4],[3,3,4],[4,4,4],[4,4,4],[4,4,4],[2,3,4]],
    [[4,4,4],[2,3,4],[3,3,4],[4,4,4],[4,4,4],[4,4,4],[2,3,4]],
]

def owas_action_category(back, arms, legs, load):
    """back ∈ {1..4}, arms ∈ {1..3}, legs ∈ {1..7}, load ∈ {1..3}"""
    return int(OWAS_AC[back-1, arms-1, legs-1, load-1])
```

---

## 4. Computation from 3D Joint Angles

Inputs: 3D coordinates of pose joints (typically hips L/R, shoulders L/R, neck/thorax,
knees L/R, ankles L/R, wrists L/R, head). Choose a world frame with +Z = up.

### 4.1 Back digit

Build:
- `pelvis = 0.5 * (hipL + hipR)`
- `thorax = 0.5 * (shoulderL + shoulderR)`  (or use C7/neck if available)
- `trunk_vec = thorax − pelvis`

**Sagittal trunk flexion** `θ_flex` =
angle between `trunk_vec` projected onto the body-sagittal plane (XZ) and the +Z axis.
Compute as `θ_flex = atan2(trunk_vec_x, trunk_vec_z)` (signed: positive forward).

**Axial trunk twist** `θ_twist` =
signed angle, in the transverse (XY) plane, between the shoulder line
`shoulderR − shoulderL` and the hip line `hipR − hipL`. Compute by projecting both
vectors to XY, normalising, and using `atan2(cross_z, dot)`.

**Lateral bend** `θ_lat` =
angle between `trunk_vec` projected onto the coronal plane (YZ) and +Z.

Decision rule:
```
bent     = (|θ_flex| > 20°)
twisted  = (|θ_twist| > 20° or |θ_lat| > 20°)
back = 1 if (not bent and not twisted)
back = 2 if (    bent and not twisted)
back = 3 if (not bent and     twisted)
back = 4 if (    bent and     twisted)
```

### 4.2 Arms digit

For each side, compute shoulder flexion/abduction:
- `upper_arm = elbow − shoulder`
- `θ_arm = angle(upper_arm, −trunk_vec)`   (180° when arm hangs straight down)

A pragmatic, robust rule (mirroring the original visual criterion "is the hand at or
above the shoulder?") is to compare `wrist.z` to `shoulder.z`:

```
aboveL = (wristL.z ≥ shoulderL.z)  or equivalently (θ_arm_L ≥ 90°)
aboveR = (wristR.z ≥ shoulderR.z)
n_above = int(aboveL) + int(aboveR)
arms = 1 if n_above == 0
arms = 2 if n_above == 1
arms = 3 if n_above == 2
```

### 4.3 Legs digit

For each side compute knee flexion (the **thigh–calf** angle, 180° = fully extended):
- `thigh = hip − knee`
- `shank = ankle − knee`
- `knee_angle = angle(thigh, shank)`

Auxiliary heuristics:
- `is_walking` ← horizontal CoM velocity, or alternating ankle separation > 0.2 *
  hip-width over time. Single-frame analyses cannot reliably set this.
- `is_kneeling` ← either knee z-coordinate close to ankle z-coordinate.
- `is_sitting` ← hip flexion ≈ 90° **and** both knees flexed, with pelvis low.

Decision pipeline (apply in this priority order):
```
if walking detected:                              legs = 7
elif kneeling detected (1 or 2 knees on ground):  legs = 6
elif sitting detected:                            legs = 1
else:
    bentL = (knee_angle_L < 150°)
    bentR = (knee_angle_R < 150°)
    unbalanced = (lateral CoM offset > 0.1 * stance_width)
    if not bentL and not bentR:
        legs = 3 if unbalanced else 2
    elif bentL and bentR:
        legs = 5 if unbalanced else 4
    else:
        # exactly one knee bent
        legs = 3
```

### 4.4 Load digit

Load is **not** derivable from pose alone — it must be supplied as metadata
(scale measurement, payload spec, or force-plate / hand-dynamometer reading).
If unknown, default to 1 and flag the observation as load-unverified.

---

## 5. Common Pitfalls and Known Confusions

1. **Back = 4 requires BOTH conditions.** Many implementations incorrectly set back = 4
   when *either* bent or twisted is true above some merged threshold. The correct
   rule is the logical AND of independent flexion and rotation tests.
2. **Arms = 2 vs 3 are sometimes swapped.** Karhu et al. (1977) define:
   2 = *one* arm at/above shoulder, 3 = *both* arms at/above shoulder. A subset of
   later texts (notably some Spanish translations) reverse the order.
3. **Knee threshold is 150°, not 90°.** The boundary between "straight" (code 2/3) and
   "bent" (code 4/5) is **150° of thigh–calf angle** (≈ 30° flexion).
4. **Legs = 1 (sitting) coexists with leg flexion.** When sitting, both knees are
   flexed (often < 150°) but legs = 1, not 4. The sitting test must precede the
   bent-knee test in the pipeline.
5. **Lateral bend and axial rotation are merged into "twist".** OWAS treats lateral
   trunk bending and axial rotation as the same digit; do not split them.
6. **20° threshold applies to back only.** It is sometimes incorrectly applied to
   shoulders or knees. Shoulder uses the shoulder-line (≈ 90° elevation), knees use
   150°.
7. **Load digit indices are 1, 2, 3 (not 0, 1, 2).** When indexing a 0-based array,
   remember to subtract 1.
8. **The leg digit "walking" is non-stationary** and cannot be diagnosed from a single
   frame; OWAS was designed for time-sampled observation (typically every 30 s).
9. **OWAS was designed for work-sampling, not single-pose scoring.** A "session" AC
   is normally derived from the frequency-weighted distribution of postures across
   the work cycle.

---

## 6. Action Category Interpretation

| AC  | Risk to the musculoskeletal system | Intervention timing |
|-----|------------------------------------|---------------------|
| 1   | Normal posture; no harmful effect on the musculoskeletal system. | **No corrective action required.** |
| 2   | Posture has some harmful effect on the musculoskeletal system. | **Corrective actions required in the near future.** |
| 3   | Posture has a distinctly harmful effect on the musculoskeletal system. | **Corrective actions required as soon as possible.** |
| 4   | Posture has an extremely harmful effect on the musculoskeletal system. | **Corrective actions required immediately.** |

For aggregated (session-level) scoring, OWAS also defines a *per-body-part* action
category, indexed by the **relative frequency** with which each digit value appears
across the sample. The Louhevaara & Suurnäkki (1992) frequency bands are:

| Relative frequency band | Translation to body-part AC |
|--------------------------|-----------------------------|
| ≤ 10 %                  | Lower-risk band             |
| > 10 % and ≤ 20 %       | Mid-low band                |
| > 20 % and ≤ 30 %       | Mid band                    |
| > 30 % and ≤ 40 %       | Mid-high band               |
| > 40 %                  | High-risk band              |

Each band maps to AC 1–4 via the body-part frequency table (Ergonautas Table 7).
Implementations should provide both a per-frame AC (§3 table) and a per-body-part AC
(frequency-based).

---

## 7. Authoritative URLs and DOIs

1. Karhu, Kansi, Kuorinka (1977), seminal paper — *Applied Ergonomics* 8(4):199–201.
   ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/0003687077901648
   PubMed: https://pubmed.ncbi.nlm.nih.gov/15677243/
2. Karhu, Härkönen, Sorvali, Vepsalainen (1981) — *Applied Ergonomics* 12(1):13–17.
   https://www.sciencedirect.com/science/article/abs/pii/0003687081900880
   PubMed: https://pubmed.ncbi.nlm.nih.gov/15676393/
3. Louhevaara V., Suurnäkki T. (1992). *OWAS: A Method for the Evaluation of Postural
   Load During Work.* Helsinki: Institute of Occupational Health.
4. Ergonautas (Universidad Politécnica de Valencia) — online reference and software:
   https://www.ergonautas.upv.es/metodos/owas/owas-ayuda.php
5. Gómez-Galán M. et al. (2017) "Musculoskeletal disorders: OWAS review."
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC5546841/
6. Gómez-Galán M. et al. (2022) "Systematic Comparison of OWAS, RULA, and REBA Based on a Literature Review."
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8744662/
7. Joshi M., Deshpande V. (2023). "An investigative sensitivity study of OWAS." *Theoretical Issues in Ergonomics Science*, 24(1).
   https://www.tandfonline.com/doi/abs/10.1080/1463922X.2022.2036859
8. Reference Python implementation: https://github.com/rs9000/ergonomics/blob/master/ergonomics/owas.py
9. ErgoIBV OWAS module: https://www.ergoibv.com/wp-content/uploads/2023/05/14__Module-PosturesOWAS.pdf
10. Bridger R.S. *Introduction to Ergonomics*, 3rd ed., CRC Press (2008).
11. Salvendy G. (Ed.) *Handbook of Human Factors and Ergonomics*, 5th ed., Wiley (2021).
