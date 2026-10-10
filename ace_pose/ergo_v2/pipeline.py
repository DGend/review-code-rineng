"""Combined OWAS + RULA + REBA pipeline on a single pose."""
from __future__ import annotations

import numpy as np

from ace_pose.major_joints import Major25Pose
from .measurability import unmeasurable_items
from .owas import score_owas
from .rula import score_rula
from .reba import score_reba
from .result import ErgoResult


def score_all(
    pose: Major25Pose,
    *,
    world_up: np.ndarray | None = None,
    load_kg: float = 0.0,
    walking: bool = False,
    static_minute: bool = False,
    repetitive_4_per_min: bool = False,
    load_repeated: bool = False,
    rapid_changes: bool = False,
    shock: bool = False,
    coupling: int = 1,
    legs_supported: bool | None = None,
    shoulder_raised: bool = False,
    upper_arm_supported: bool = False,
    wrist_deviation: bool = False,
    process_id: int | None = None,
    process_label: str | None = None,
    domain: str | None = None,
    bilateral: bool = True,
) -> ErgoResult:
    """Run OWAS, RULA, and REBA on the same pose and return a unified result.

    Args:
        pose: Major-25 keypoint pose.
        world_up: world up-axis. Defaults to body-up fallback.
        load_kg: handled load (kg).
        walking: caller-supplied flag for OWAS leg digit 7.
        static_minute, repetitive_4_per_min, rapid_changes: temporal activity
            flags used by RULA muscle-use and REBA activity score.
        shock: shock / sudden-force flag.
        coupling: REBA coupling 0..3.
        legs_supported: explicit override for the "supported" flag.
        shoulder_raised, upper_arm_supported, wrist_deviation: optional
            posture-detail flags propagated to RULA and REBA.
        process_id, process_label, domain: metadata for downstream calibration.
        bilateral: if True, score both sides separately and keep the worse
            side as the headline; if False, only score the right side.
    """
    # Angles are measured in the image plane; the depth coordinate is dropped
    # so it cannot leak into segment lengths or joint angles (angles.py).
    kp = np.array(pose.keypoints, dtype=float)
    kp[:, 2] = 0.0
    pose = Major25Pose(keypoints=kp, confidence=pose.confidence, source=pose.source)
    owas = score_owas(pose, load_kg=load_kg, walking=walking, world_up=world_up)

    rula_kwargs = dict(
        world_up=world_up, load_kg=load_kg,
        static_minute=static_minute,
        repetitive_4_per_min=repetitive_4_per_min,
        load_repeated=load_repeated,
        shock=shock,
        legs_supported=legs_supported,
        shoulder_raised=shoulder_raised,
        upper_arm_supported=upper_arm_supported,
        wrist_deviation=wrist_deviation,
    )
    reba_kwargs = dict(
        world_up=world_up, load_kg=load_kg, shock=shock,
        coupling=coupling,
        activity_static=static_minute,
        activity_repetitive=repetitive_4_per_min,
        activity_rapid=rapid_changes,
        legs_supported=legs_supported,
        shoulder_raised=shoulder_raised,
        upper_arm_supported=upper_arm_supported,
        wrist_deviation=wrist_deviation,
    )

    unmeasurable = unmeasurable_items(pose)
    if bilateral:
        rl = score_rula(pose, side="left", **rula_kwargs)
        rr = score_rula(pose, side="right", **rula_kwargs)
        bl = score_reba(pose, side="left", **reba_kwargs)
        br = score_reba(pose, side="right", **reba_kwargs)
        # keep the worse side among the arms that are seen (both if neither is)
        sides = [s for s in ("right", "left") if f"arm_{s}" not in unmeasurable] or ["right", "left"]
        ru = {"left": rl, "right": rr}
        re = {"left": bl, "right": br}
        rula = max((ru[s] for s in sides), key=lambda r: r.grand_score)
        reba = max((re[s] for s in sides), key=lambda r: r.grand_score)
        return ErgoResult(
            owas=owas, rula=rula, reba=reba,
            rula_left=rl, rula_right=rr,
            reba_left=bl, reba_right=br,
            process_id=process_id, process_label=process_label, domain=domain,
            extras={"unmeasurable": unmeasurable},
        )

    rula = score_rula(pose, side="right", **rula_kwargs)
    reba = score_reba(pose, side="right", **reba_kwargs)
    return ErgoResult(
        owas=owas, rula=rula, reba=reba,
        process_id=process_id, process_label=process_label, domain=domain,
        extras={"unmeasurable": unmeasurable},
    )

