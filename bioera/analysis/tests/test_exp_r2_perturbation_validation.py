# PWW-DISCLOSURE: public
"""Unit tests for the deterministic helpers of exp_r2_perturbation_validation (pytest, conda env bioera_mirror_test).
Every expected value below is worked out by hand in the comment next to it."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import exp_r2_perturbation_validation as P  # noqa: E402
from ace_pose.major_joints import Major25Pose  # noqa: E402


def test_normalized_error_scales_by_torso_over_ref_and_masks_missing():
    res = np.array([[[0.1, 0.2], [0.3, 0.4]]])                   # (1 image, 2 joints, xy) in predicted-torso units
    out = P.normalized_error(res, np.array([2.0]), np.array([4.0]), np.array([[True, False]]))
    np.testing.assert_allclose(out[0, 0], [0.05, 0.10])         # x (2 / 4) = x 0.5
    assert np.isnan(out[0, 1]).all()                            # joint not annotated


def test_robust_sd_and_kurtosis_by_hand():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert abs(P.robust_sd(x) - 2 / 1.349) < 1e-9               # linear-interpolated q25 = -1, q75 = 1 -> IQR 2
    assert abs(P.excess_kurtosis(np.array([-1.0, 1.0])) - (-2.0)) < 1e-12   # m2 = 1, m4 = 1 -> 1 - 3


def test_rank_corr_with_ties_and_nan():
    assert abs(P.rank_corr([1, 2, 3], [1, 3, 2]) - 0.5) < 1e-12  # ranks (1,2,3) vs (1,3,2): 1 - 6*2/24 = 0.5
    assert abs(P.rank_corr([1, 2, 3, np.nan], [3, 2, 1, 9]) + 1.0) < 1e-12   # NaN pair dropped, perfectly reversed


def test_class_probs_prefix_and_mc_se():
    lev = np.array([[2, 2, 3, 2, -1, 3, 3, 3]], dtype=np.int8)   # classes 1..3, one failed sample
    p4, t4 = P.class_probs(lev, 4, 1, 3)                         # first 4: three 2s, one 3
    np.testing.assert_allclose(p4[0], [0, 0.75, 0.25])
    p8, t8 = P.class_probs(lev, 8, 1, 3)                         # 7 valid: three 2s, four 3s
    assert t8[0] == 7 and abs(p8[0, 2] - 4 / 7) < 1e-12
    assert abs(P.mc_se(np.array([0.5]), np.array([100.0]))[0] - 0.05) < 1e-12   # sqrt(.25 / 100)


def test_col_stats_kinds():
    Q = np.array([[1.0, 1.0, 1.0, np.nan], [3.0, 3.0, 3.0, 2.0], [5.0, 5.0, 5.0, 4.0]])
    out = P.col_stats(Q, ["mean", "median", "rms", "n"])
    np.testing.assert_allclose(out, [3.0, 3.0, np.sqrt((1 + 9 + 25) / 3), 2.0])


def test_angles_of_hand_built_pose():
    kp = np.zeros((25, 3))
    kp[5], kp[6] = [-5, -10, 0], [5, -10, 0]                     # shoulders (image y down, up is -y)
    kp[11], kp[12] = [-5, 0, 0], [5, 0, 0]                       # hips
    kp[23], kp[24] = [0, -10, 0], [0, 0, 0]                      # neck, pelvis: upright trunk
    kp[8], kp[10] = [5, -5, 0], [10, -5, 0]                      # right elbow below shoulder, wrist sideways -> 90 degree bend
    kp[14], kp[16] = [5, 5, 0], [5, 10, 0]                       # right hip -> knee -> ankle in a straight line
    pose = Major25Pose(keypoints=kp, confidence=np.ones(25), source=np.zeros(25, np.int8))
    a = dict(zip(P.ANG, P.angles_of(pose)))
    assert abs(a["trunk"]) < 1e-6                                # trunk vertical
    assert abs(a["forearm_r"] - 90.0) < 1e-6                     # interior angle 90 -> flexion 180 - 90
    assert abs(a["knee_r"]) < 1e-6                               # straight leg -> flexion 0


def test_wrapped_std_and_xy_only():
    a = np.array([[179.0], [-179.0]])                           # 2 degrees apart across the +-180 seam
    assert abs(P.wrapped_std(a, np.array([180.0]))[0] - 1.0) < 1e-9   # deviations -1 and +1 -> sd 1
    kp = np.zeros((25, 3))
    kp[:, 2] = 7.0
    base = Major25Pose(keypoints=kp, confidence=np.ones(25), source=np.zeros(25, np.int8))
    pert = Major25Pose(keypoints=kp + 1.0, confidence=np.ones(25), source=np.zeros(25, np.int8))
    out = P.xy_only([pert], base)[0].keypoints
    assert (out[:, 2] == 7.0).all() and (out[:, :2] == 1.0).all()   # z restored, x y kept perturbed
    k2 = np.zeros((25, 3)); k2[0], k2[1] = [0, 0, 0], [10, 4, 2]
    assert abs(P.z_range_ratio(k2) - 0.2) < 1e-12              # z range 2 / max(x range 10, y range 4)
