# PWW-DISCLOSURE: public
"""Figure 13: pipeline vs. independent-expert action levels on the 22 pilot
images, as submitted and after the corrections of Section 4.9.

Source data (public, derived): pww_data.resolve("expert_pilot_22"), the
numeric table built from the expert's workbook by the private script
build_expert_pilot_table.py. Each (image, standard) has a "system_submitted"
row (the pipeline score sent to the expert) and an "expert" row. The
expert's free-text reasons are withheld; the mechanism flags derived from
them by keyword match (walking, load, sustained/repeated activity) are
included.

Writes:
  ace_pose/_runs/expert_pilot/expert_vs_pipeline.csv  (tidy, per id x standard)
  topic/E/paper/figures/fig_expert_comparison.png/.pdf (2 x 3 confusion matrices;
  the bottom row reads the final stage of load_fix_before_after.csv)
Prints the summary numbers that go into manuscript Section 4.9, so
prose and figure cannot drift apart.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pww_data import resolve

OUT_CSV_DIR = ROOT / "ace_pose/_runs/expert_pilot"
OUT_FIG_DIR = Path(__file__).resolve().parents[1] / "paper/figures"
MECHANISMS = {"legs_walking": "mech_legs_walking", "load": "mech_load",
              "activity_repetition": "mech_activity"}


def direction(system: int, expert: int) -> str:
    if system < expert:
        return "under"
    if system > expert:
        return "over"
    return "match"


def main():
    scores = pd.read_csv(resolve("expert_pilot_22"))
    rows = []
    for (id_, standard), g in scores.groupby(["id", "standard"], sort=False):
        s = g[g.rater == "system_submitted"].iloc[0]
        e = g[g.rater == "expert"].iloc[0]
        rows.append({
            "id": id_,
            "domain": s.domain,
            "standard": standard,
            "system_action": int(s.action),
            "expert_action": int(e.action),
            "exact_match": int(s.action) == int(e.action),
            "direction": direction(int(s.action), int(e.action)),
            "mechanisms": ",".join(m for m, col in MECHANISMS.items() if s[col]),
        })

    tidy = pd.DataFrame(rows)
    OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(OUT_CSV_DIR / "expert_vs_pipeline.csv", index=False)

    print("=== Exact-match counts (of n=22) ===")
    for standard in ["OWAS", "RULA", "REBA"]:
        sub = tidy[tidy.standard == standard]
        n_match = int(sub.exact_match.sum())
        print(f"{standard}: {n_match}/22 exact match")

    print("\n=== Directional bias (system vs expert action level) ===")
    for standard in ["OWAS", "RULA", "REBA"]:
        sub = tidy[tidy.standard == standard]
        counts = sub["direction"].value_counts()
        n = len(sub)
        under = counts.get("under", 0)
        match = counts.get("match", 0)
        over = counts.get("over", 0)
        print(f"{standard}: under={under}/{n} ({100*under/n:.1f}%)  "
              f"match={match}/{n} ({100*match/n:.1f}%)  "
              f"over={over}/{n} ({100*over/n:.1f}%)")

    print("\n=== Mechanism attribution (keyword flags derived from the withheld reason text, RULA+REBA+OWAS disagreements combined, counts not mutually exclusive) ===")
    disagree = tidy[~tidy.exact_match]
    for mech in MECHANISMS:
        n = disagree["mechanisms"].fillna("").apply(lambda s: mech in s.split(",")).sum()
        print(f"{mech}: {n} disagreeing rows mention this")
    n_unclassified = disagree["mechanisms"].fillna("").apply(lambda s: s == "").sum()
    print(f"unclassified (no keyword matched): {n_unclassified}")

    print("\n=== E5 / E10 flag ===")
    for sid in ["E5", "E10"]:
        sub = tidy[tidy.id == sid]
        for _, r in sub.iterrows():
            print(f"{sid} {r.standard}: system={r.system_action} expert={r.expert_action} "
                  f"mechanisms={r.mechanisms!r}")

    # ---- figure: confusion matrices, system (rows) vs expert (columns), as submitted
    # (top) and after every correction of Section 4.9 (bottom, fig_expert_load_fix.py) ----
    stages = pd.read_csv(OUT_CSV_DIR / "load_fix_before_after.csv")
    final = stages.set_index(["id", "standard"]).taskload
    levels_of = {"OWAS": [1, 2, 3, 4], "RULA": [1, 2, 3, 4], "REBA": [0, 1, 2, 3, 4]}
    cmap = LinearSegmentedColormap.from_list("n", ["#ffffff", "#2b6cb0"])
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.4))
    for row, (label, system) in enumerate([
            ("Scores as submitted to the expert", lambda r: r.system_action),
            ("After the corrections (development images)", lambda r: int(final[(r.id, r.standard)]))]):
        for col, standard in enumerate(["OWAS", "RULA", "REBA"]):
            ax, lv = axes[row, col], levels_of[standard]
            sub = tidy[tidy.standard == standard]
            mat = np.zeros((len(lv), len(lv)), int)
            for r in sub.itertuples():
                mat[lv.index(system(r)), lv.index(r.expert_action)] += 1
            ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(8, mat.max()))
            for a in range(len(lv)):
                ax.add_patch(plt.Rectangle((a - .5, a - .5), 1, 1, fill=False, lw=2, ec="#1a202c"))
                for b in range(len(lv)):
                    if mat[a, b]:
                        ax.text(b, a, mat[a, b], ha="center", va="center", fontsize=12, fontweight="bold",
                                color="white" if mat[a, b] > 5 else "#1a202c")
            lower, higher = int(np.triu(mat, 1).sum()), int(np.tril(mat, -1).sum())
            print(f"{label} | {standard}: exact {int(np.trace(mat))}/22, system lower {lower}, higher {higher}")
            ax.set_xticks(range(len(lv)), lv)
            ax.set_yticks(range(len(lv)), lv)
            ax.set_xlabel("Expert action level")
            ax.set_ylabel("System action level")
            ax.set_title(f"{standard}: exact {int(np.trace(mat))}/22\nsystem lower {lower}, higher {higher}", fontsize=11)
        axes[row, 0].annotate(label, xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90,
                              va="center", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.01, "Cells above the diagonal: system lower than the expert; below: system higher. "
             "Outlined diagonal: exact agreement.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0.03, 0.03, 1, 1])
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_DIR / "fig_expert_comparison.png", dpi=200)
    fig.savefig(OUT_FIG_DIR / "fig_expert_comparison.pdf")
    print(f"\nwrote {OUT_FIG_DIR / 'fig_expert_comparison.png'}")


if __name__ == "__main__":
    main()
