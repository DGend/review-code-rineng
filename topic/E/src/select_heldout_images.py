# PWW-DISCLOSURE: public
"""Freeze a held-out image set for the second expert-scoring round, before any rule change.

The 22 pilot images were used to find the scorer's problems, so they cannot also
test the fixes. This draws a new set the scorer has never been tuned on:
  - every recording session that contains a pilot image is excluded;
  - only images whose analysed person matches the labelled worker are eligible;
  - 12 images per domain, at most MAX_PER_SESSION per session, spread over processes
    (round-robin over processes, random within each, fixed seed).
The list and its SHA-256 are written once; re-running refuses to overwrite it.

Usage: python topic/E/src/select_heldout_images.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "topic/E/result/heldout_selection"
SEED, PER_DOMAIN = 20261008, 12
MAX_PER_SESSION = {"eldercare": 2, "logistics": 1, "manufacturing": 1}


def session(stem: pd.Series) -> pd.Series:
    return stem.str.rsplit("_", n=1).str[0]


def pick(g: pd.DataFrame, n: int, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    """Round-robin over processes; within a process take a random image from a session not yet full."""
    pools = {p: list(rng.permutation(x.index)) for p, x in g.groupby("process_label")}
    order = list(rng.permutation(sorted(pools)))
    taken, per_session = [], {}
    while len(taken) < n and any(pools.values()):
        for p in order:
            while pools[p]:
                i = pools[p].pop()
                s = g.at[i, "session"]
                if per_session.get(s, 0) < cap:
                    taken.append(i)
                    per_session[s] = per_session.get(s, 0) + 1
                    break
            if len(taken) == n:
                break
    if len(taken) < n:
        sys.exit(f"only {len(taken)} eligible images for {g.domain.iloc[0]}")
    return g.loc[taken]


def main():
    csv = OUT / "heldout_36.csv"
    if csv.exists():
        sys.exit(f"{csv} already exists; the held-out set is frozen")
    d = pd.read_csv(ROOT / "ace_pose/_runs/batch_v1/summary.csv")
    pilot = pd.read_csv(ROOT / "datasets/public/expert_pilot_22/expert_pilot_scores.csv").drop_duplicates("id")
    d["session"] = session(d.file_stem)
    eligible = d[~d.session.isin(set(session(pilot.file_stem))) & d.label_match_ok.astype(bool)
                 & (d.status == "ok")]
    rng = np.random.default_rng(SEED)
    sel = pd.concat([pick(g, PER_DOMAIN, MAX_PER_SESSION[dom], rng) for dom, g in eligible.groupby("domain")])
    sel = sel[["file_stem", "domain", "factory_id", "process_label", "session"]].reset_index(drop=True)
    sel.insert(0, "id", [f"H{i + 1:02d}" for i in range(len(sel))])
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_csv(csv, index=False)
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    (OUT / "heldout_36.sha256").write_text(f"{digest}  heldout_36.csv\n")
    print(sel.groupby(["domain", "process_label"]).size().to_string())
    print(f"sessions: {sel.session.nunique()}  sha256 {digest}")


if __name__ == "__main__":
    main()
