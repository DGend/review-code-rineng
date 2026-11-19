# PWW-DISCLOSURE: public
"""Forward kinematics of BVH motion files (CarDA Xsens export)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carda_bvh import load_bvh  # noqa: E402

BVH = """HIERARCHY
ROOT Hips
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
 JOINT Chest
 {
  OFFSET 0 10 0
  CHANNELS 3 Yrotation Xrotation Zrotation
  JOINT Head
  {
   OFFSET 0 0 5
   CHANNELS 3 Yrotation Xrotation Zrotation
   End Site
   {
    OFFSET 0 1 0
   }
  }
 }
}
MOTION
Frames: 99
Frame Time: 0.016667
1 2 3 0 0 0 0 0 0 0 0 0
0 0 0 0 0 90 0 0 0 0 0 0
0 0 0 0 0 0 90 0 0 0 0 0
"""


def _load(tmp_path):
    p = tmp_path / "t.bvh"
    p.write_text(BVH)
    return load_bvh(p)


def test_joint_names_and_real_frame_count(tmp_path):
    names, pos, fps = _load(tmp_path)
    assert names == ["Hips", "Chest", "Head"]
    assert pos.shape == (3, 3, 3)          # the header's "Frames: 99" is ignored, data rows count
    assert fps == 60


def test_translation(tmp_path):
    _, pos, _ = _load(tmp_path)
    assert np.allclose(pos[0], [[1, 2, 3], [1, 12, 3], [1, 12, 8]])


def test_root_z_rotation_turns_children(tmp_path):
    _, pos, _ = _load(tmp_path)
    assert np.allclose(pos[1], [[0, 0, 0], [-10, 0, 0], [-10, 0, 5]], atol=1e-9)


def test_child_rotation_applies_to_grandchild_only(tmp_path):
    _, pos, _ = _load(tmp_path)                       # Chest Yrotation 90: z -> x
    assert np.allclose(pos[2], [[0, 0, 0], [0, 10, 0], [5, 10, 0]], atol=1e-9)


def test_global_rotations_are_returned_on_request(tmp_path):
    p = tmp_path / "t.bvh"
    p.write_text(BVH)
    names, pos, fps, rot = load_bvh(p, with_rotations=True)
    assert rot.shape == (3, 3, 3, 3)
    assert np.allclose(rot[1, 0] @ [0, 10, 0], pos[1, 1] - pos[1, 0])     # root Z 90 turns the chest offset
    assert np.allclose(rot[2, 1] @ [0, 0, 5], pos[2, 2] - pos[2, 1])      # chest Y 90 turns the head offset
