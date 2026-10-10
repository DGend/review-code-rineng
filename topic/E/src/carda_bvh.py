# PWW-DISCLOSURE: public
"""Global joint positions from a BVH motion file (forward kinematics).

Written for the CarDA Xsens exports (Y up, millimetres, rotation channels in Y X Z order),
but reads the channel order of each joint from the file. The "Frames:" header of the CarDA
files does not match the data (e.g. 19,481 declared, 3,461 rows), so frames are counted
from the data rows.

    names, pos, fps = load_bvh(path)   # pos: (frames, joints, 3), End Sites left out
    names, pos, fps, rot = load_bvh(path, with_rotations=True)   # rot: (frames, joints, 3, 3) global
"""
from pathlib import Path

import numpy as np


def _rot(axis: str, deg: np.ndarray) -> np.ndarray:
    """Rotation matrices (n, 3, 3) about one axis."""
    a = np.radians(deg)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    m = {"X": [[o, z, z], [z, c, -s], [z, s, c]],
         "Y": [[c, z, s], [z, o, z], [-s, z, c]],
         "Z": [[c, -s, z], [s, c, z], [z, z, o]]}[axis]
    return np.moveaxis(np.array(m), -1, 0)


def load_bvh(path, with_rotations=False):
    text = Path(path).read_text()
    head, motion = text.split("MOTION", 1)
    names, parents, offsets, channels, stack, skip = [], [], [], [], [], 0
    tok = head.split()
    i = 0
    while i < len(tok):
        t = tok[i]
        if t in ("ROOT", "JOINT"):
            names.append(tok[i + 1])
            parents.append(stack[-1] if stack else -1)
            i += 2
        elif t == "End":                       # End Site: parse past its block, not a joint
            skip += 1
            i += 2
        elif t == "{":
            stack.append(None if skip else len(names) - 1)
            i += 1
        elif t == "}":
            if stack.pop() is None:
                skip -= 1
            i += 1
        elif t == "OFFSET":
            if not skip:
                offsets.append([float(x) for x in tok[i + 1:i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(tok[i + 1])
            channels.append(tok[i + 2:i + 2 + n])
            i += 2 + n
        else:
            i += 1
    lines = motion.strip().split("\n")
    fps = round(1 / float(next(l for l in lines if l.startswith("Frame Time")).split()[-1]))
    width = sum(len(c) for c in channels)
    data = np.array([[float(x) for x in l.split()] for l in lines if len(l.split()) == width])
    nf, nj = len(data), len(names)
    pos, rot, col = np.zeros((nf, nj, 3)), np.zeros((nf, nj, 3, 3)), 0
    for j in range(nj):
        local_t, local_r = np.tile(offsets[j], (nf, 1)), np.tile(np.eye(3), (nf, 1, 1))
        for ch in channels[j]:
            v = data[:, col]
            col += 1
            if ch.endswith("position"):
                local_t[:, "XYZ".index(ch[0])] += v
            else:
                local_r = local_r @ _rot(ch[0], v)
        p = parents[j]
        if p < 0:
            pos[:, j], rot[:, j] = local_t, local_r
        else:
            pos[:, j] = pos[:, p] + np.einsum("nab,nb->na", rot[:, p], local_t)
            rot[:, j] = rot[:, p] @ local_r
    return (names, pos, fps, rot) if with_rotations else (names, pos, fps)
