# Joint range of motion (ROM) used by the 3D body model

`ace_pose/lift3d_rom.py` bounds every joint angle of its body model by the values below. One
set of values is used for every worker, regardless of sex and age (project decision,
2026-10-09).

## Source

US Department of Veterans Affairs, Schedule for Rating Disabilities, 38 CFR 4.71 (Plates I and
II) and 4.71a (General Rating Formula for Diseases and Injuries of the Spine, Note 2), 7-1-24
edition (GovInfo, CFR-2024-title38-vol1, sections 4.71 and 4.71a; eCFR text read 2026-10-09).
It states one normal range per motion. Note 3 of 4.71a: an examiner may accept a different
range as normal for an individual "because of age, body habitus, neurologic disease, or other
factors"; we do not use that exception.

Verbatim (4.71a, Note 2): "normal forward flexion of the cervical spine is zero to 45 degrees,
extension is zero to 45 degrees, left and right lateral flexion are zero to 45 degrees, and
left and right lateral rotation are zero to 80 degrees. Normal forward flexion of the
thoracolumbar spine is zero to 90 degrees, extension is zero to 30 degrees, left and right
lateral flexion are zero to 30 degrees, and left and right lateral rotation are zero to 30
degrees."

Plates I and II (read from the figures): shoulder forward elevation 0-180, abduction 0-180,
external and internal rotation 0-90 each; elbow flexion 0-145; forearm pronation 0-80,
supination 0-85; wrist extension 0-70, palmar flexion 0-80, radial deviation 0-20, ulnar
deviation 0-45; hip flexion 0-125, abduction 0-45; knee flexion 0-140; ankle dorsiflexion
0-20, plantar flexion 0-45.

## Model parameters

Signs: flexion forward, abduction away from the midline. Lower bound = extension/adduction
side (negative), upper bound = flexion/abduction side.

| Parameter | Lower | Upper | Source |
|---|---:|---:|---|
| trunk_flex (thoracolumbar, relative to the pelvis) | -30 | 90 | 4.71a Note 2 |
| trunk_lat | -30 | 30 | 4.71a Note 2 |
| trunk_twist | -30 | 30 | 4.71a Note 2 |
| neck_flex | -45 | 45 | 4.71a Note 2 |
| neck_lat | -45 | 45 | 4.71a Note 2 |
| neck_rot | -80 | 80 | 4.71a Note 2 |
| shoulder_flex | -60 | 180 | upper: Plate I; lower (extension): **no value in the source, assumption** |
| shoulder_abd | -30 | 180 | upper: Plate I; lower (adduction across the body): **assumption** |
| shoulder_rot | -90 | 90 | Plate I |
| elbow | 0 | 145 | Plate I (no hyperextension) |
| forearm_rot (negative = supination) | -85 | 80 | Plate I (0 = mid-position, thumb forward with the arm hanging) |
| wrist_flex (negative = extension) | -70 | 80 | Plate I |
| wrist_dev (negative = radial) | -20 | 45 | Plate I |
| hip_flex | -30 | 125 | upper: Plate II; lower (extension): **assumption** |
| hip_abd | -30 | 45 | upper: Plate II; lower (adduction): **assumption** |
| hip_rot | -45 | 45 | **assumption** (not in the source) |
| knee | 0 | 140 | Plate II (no hyperextension) |
| ankle (negative = plantar flexion) | -45 | 20 | Plate II |

Not joints, so not ROM: root_yaw (heading, free), root_pitch -30..120 and root_roll -30..30
(orientation of the pelvis in the world; model assumptions that keep the body off the ground
plane only loosely).

## Hand geometry

Hand points fitted to the 2D hand keypoints: the middle, index and little-finger knuckles (MCP)
at 0.06, 0.058 and 0.054 of stature from the wrist, the index and little finger 0.022 to either
side (model geometry, assumption; hand length 0.108 of stature after Winter). Ground-truth wrist
angles are taken with the same geometry from the motion-capture wrist rotation, so the
geometry assumption does not bias the comparison.

## Standing-balance prior (optional, `fit(balance=True)`)

Added 2026-10-10 after reconstructions whose centre of mass lay up to 0.31 of stature off the feet.
The camera's pitch (+-30 degrees) and roll (+-20 degrees) relative to gravity are estimated, and two
soft penalties keep the whole-body centre of mass (segment masses and centre-of-mass positions after
Dempster, as tabulated by Winter) above the centre of the feet and both feet at floor level. It is a
prior for standing work: a held load or walking legitimately moves the centre of mass off the feet.

## Open

The four assumption rows (shoulder extension and cross-body adduction, hip extension, hip
adduction, hip rotation) need a source that gives one value for all sexes and ages. The
CDC reference study (Soucie et al., Haemophilia 2010) measured hip extension but reports it by
sex and age group only. Candidates not yet read in the original: AAOS, Joint Motion: Method of
Measuring and Recording (1965); the Korean industrial-accident disability rating rules.
