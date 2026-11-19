# PWW-DISCLOSURE: public
"""Recompute the paper's verifiable numbers from the result files and check them.

Two modes, chosen automatically:
- manuscript mode (the manuscript sources are present): every recomputed value must
  appear verbatim in the manuscript or response letter. `--write-expected` then
  records the checked values in paper_numbers_expected.json.
- release mode (manuscript sources absent, e.g. the review mirror): every recomputed
  value must equal the recorded value in paper_numbers_expected.json; each entry
  names the manuscript location it comes from.

Analysis scripts whose output is only printed are re-run here, so the check
also exercises them. Sections whose inputs are absent are reported as SKIP with
the reason. Exit code 1 if any check fails.

Usage: python bioera/analysis/src/verify_manuscript_numbers.py [--write-expected]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parent
E = ROOT / "bioera/analysis"
BATCH = ROOT / "ace_pose/_runs/batch_v1"
MS = E / "paper/manuscript"
LETTER = E / "paper/build/BIOERA_RiE_response_to_reviewers.md"
EXPECTED = SRC / "paper_numbers_expected.json"
REPRO = E / "result/reproduce"
PY = sys.executable
# Test files of the retired v1 scorer, not part of the released package.
LEGACY_TESTS = ["test_pipeline.py", "test_end_to_end.py", "test_gold_lifter.py"]
# reproduce.sh writes bioera/analysis/result/reproduce/<key>/metrics.json; the dated folders are
# the archived runs the manuscript was checked against.
ARCHIVE = {
    "exp005": "bioera/analysis/result/exp-005-per-domain-ece-mcfix-20261005",
    "exp010": "bioera/analysis/result/exp-010-worker-cluster-tost-mcfix-20261005",
    "exp011": "bioera/analysis/result/exp-011-worker-cluster-ece-deterministic-20261006",
    "exp012": "bioera/analysis/result/exp-012-worker-holdout-refit-mcfix-20261005",
    "exp001c": "bioera/analysis/result/exp-001c-shapley-hardening-legsfix-20261005",
    "exp006": "bioera/analysis/result/reproduce/exp006",
}

MANUSCRIPT_MODE = MS.is_dir() and LETTER.is_file()
if MANUSCRIPT_MODE:
    TEXT = {p.name: p.read_text(encoding="utf-8") for p in sorted(MS.glob("*.md"))}
    TEXT["letter"] = LETTER.read_text(encoding="utf-8")
    EXP = {}
else:
    if not EXPECTED.is_file():
        sys.exit(f"release mode needs {EXPECTED.name}")
    EXP = {(e["section"], e["i"]): e for e in json.loads(EXPECTED.read_text())}
rows, skipped = [], []
_sec, _i = "", 0


def section(name, needs=()):
    """Start a section; return False (and record SKIP) if an input is missing."""
    global _sec, _i
    _sec, _i = name, 0
    missing = [str(p.relative_to(ROOT)) for p in needs if not p.exists()]
    if missing:
        skipped.append((name, "missing " + ", ".join(missing)))
        return False
    return True


def check(where, needle, value, ok=True):
    """Manuscript mode: `needle` must appear in `where`. Release mode: `needle` must
    equal the recorded needle for this position. `ok` is an extra numeric verdict."""
    global _i
    _i += 1
    if MANUSCRIPT_MODE:
        good = ok and needle in TEXT[where]
    else:
        e = EXP.get((_sec, _i))
        good = ok and e is not None and e["needle"] == needle and e["where"] == where
    rows.append(dict(section=_sec, i=_i, where=where, needle=needle, value=str(value),
                     verdict="MATCH" if good else "MISMATCH"))


W = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen".split()  # number words as the manuscript spells them


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"[{100*lo:.1f}, {100*hi:.1f}]"


def metrics(key):
    for p in (REPRO / key / "metrics.json", ROOT / ARCHIVE[key] / "metrics.json"):
        if p.is_file():
            return json.loads(p.read_text())
    raise FileNotFoundError(key)


def has_metrics(*keys):
    return [REPRO / k / "metrics.json" if (REPRO / k).exists()
            else ROOT / ARCHIVE[k] / "metrics.json" for k in keys]


_runs = {}


def run(script):
    """Run an analysis script once and return its stdout (fails loudly on error)."""
    if script not in _runs:
        r = subprocess.run([PY, str(SRC / f"{script}.py")], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{script}.py failed:\n{r.stderr[-1500:]}")
        _runs[script] = r.stdout
    return _runs[script]


# --- cohort composition -------------------------------------------------------
s = pd.read_csv(BATCH / "summary.csv")
s = s[s.status == "ok"]
dom = s.factory_id.map({1: "m", 2: "m", 3: "m", 4: "m", 5: "m", 6: "e", 7: "l"})
section("cohort")
check("09.supplementary.md", "**6,007**", f"{len(s):,}", len(s) == 6007)
for fid, n in s.factory_id.value_counts().sort_index().items():
    check("09.supplementary.md", f"| {n:,} |", f"F{fid} {n:,}")
proc = s.process_id.value_counts()
check("09.supplementary.md", "Sasang-1 (n = 1,947", f"{proc[1]:,}", proc[1] == 1947)
check("09.supplementary.md", "Weld-1 (n = 619", f"{proc[7]}", proc[7] == 619)
px = s.groupby(["img_w", "img_h"]).factory_id.agg(lambda x: sorted(set(x)))
big = s[(s.img_w == 3840) & (s.img_h == 2160)]
check("03.methods.md", f"(3840 x 2160 at F1-F5; 2704 x 1520 or 1920 x 1080 at F6 and F7)", f"3840x2160 on F{sorted(set(big.factory_id))}, {len(big):,} images; others {px.drop((3840, 2160)).to_dict()}",
      sorted(set(big.factory_id)) == [1, 2, 3, 4, 5] and len(big) == 4001 and set(px.index) == {(3840, 2160), (2704, 1520), (1920, 1080)}
      and all(set(v) <= {6, 7} for k, v in px.items() if k != (3840, 2160)))
check("09.supplementary.md", "Images were collected for 16", str(s.process_id.nunique()), s.process_id.nunique() == 16)
check("00.abstract.md", f"({len(s):,} images across {W[s.factory_id.nunique()]} Korean workplaces", "data availability paragraph")
rg = s.groupby(dom).factory_id.agg(["min", "max"])
fr = {k: f"F{r['min']}" + (f"-F{r['max']}" if r["max"] != r["min"] else "") for k, r in rg.iterrows()}
envs = f"(Manufacturing {fr['m']}, Eldercare {fr['e']}, Logistics {fr['l']})"
check("04.results.md", f"high-risk pose from each of the {W[s.factory_id.nunique()]} Busan workplaces (F", "Fig. 3 text: workplaces")
check("04.results.md", f"One representative high-risk pose from each of the {W[s.factory_id.nunique()]} workplaces (F{s.factory_id.min()}-F{s.factory_id.max()})", "Fig. 3 caption")
check("04.results.md", f"across three risk tiers per work environment {envs}", "Fig. 4 text: domain ranges")
check("04.results.md", f"across the three work domains {envs}", "Fig. 6 caption: domain ranges")
check("06.conclusion.md", f"with {W[s[s.factory_id == 6].worker_id.nunique()]} caregivers at one site", "eldercare workers")
# Fig. 3 text: 133 WholeBody keypoints minus 42 hand points; the retained groups follow the COCO-WholeBody layout
n133 = np.load(next((BATCH / "details").glob("*.npz")))["kpts3d_133"].shape[0]
n_body, n_feet, n_face = 17, 6, 68
check("04.results.md", f"overlaying the {n_body + n_feet + n_face} retained RTMW3D keypoints (body {n_body} + feet {n_feet} + face {n_face} for head-reference visualisation; hands {n133 - n_body - n_feet - n_face} excluded)",
      "retained keypoints", n133 == 133)
sys.path.insert(0, str(ROOT))
from ace_pose.disc_load import NIOSH_AL_N, NIOSH_MPL_N  # noqa: E402
check("04.results.md", f"below the Action Limit of {NIOSH_AL_N:,.0f} N", "NIOSH AL")
check("04.results.md", f"at or above the Maximum Permissible Limit of {NIOSH_MPL_N:,.0f} N", "NIOSH MPL")

# --- standards cross-check (Section 4.2, Table 1) ------------------------------
section("standards")
sys.path.insert(0, str(ROOT / "bioera/core_experiments/src/public"))
import exp002_standards_audit as audit  # noqa: E402
ow = audit.diff_owas()
total = sum(f()["total_cells"] for f in (audit.diff_owas, audit.diff_rula_A, audit.diff_rula_B, audit.diff_rula_C,
                                         audit.diff_reba_A, audit.diff_reba_B, audit.diff_reba_C))
check("03.methods.md", f"({total} cells; Section 4.2", "cells audited")
check("04.results.md", f"| OWAS Action Category ({ow['total_cells']} cells) | {ow['wrong_cells']} |", "OWAS cells differing")
check("04.results.md", f"covering all {ow['total_cells']} OWAS Action-Category cells, all {audit.diff_rula_A()['total_cells']} RULA Table-A cells, and the RULA Table B/C and REBA Table A/B/C cells ({total} cells in total)",
      "Table 3 paragraph: audited cells")
check("04.results.md", f"The {ow['wrong_cells']} OWAS cells (back 1 / arms 3 / legs 7", "OWAS cells still differing")
codes = {tuple(d["code"]) for d in ow["diffs"]}
n_in = int(s[["owas_back", "owas_arms", "owas_legs", "owas_load"]].apply(tuple, axis=1).isin(codes).sum())
check("04.results.md", "None (no cohort pose falls in these cells)", f"{n_in} cohort poses in the differing OWAS cells", n_in == 0)

# --- domain statistics and equivalence (Section 4.3, letter) ------------------
if section("domains", has_metrics("exp010")):
    for col, tag in [("rula_grand", "RULA H = "), ("reba_grand", "REBA H = ")]:
        H, _ = kruskal(*[s[col][dom == k] for k in "mel"])
        check("letter", f"{tag}{H:.1f}", f"H={H:.1f} eps2={H/(len(s)-1):.4f}")
        check("letter", f"{H/(len(s)-1):.4f}", f"eps2 {col}")
    reba = s.groupby(dom).reba_grand.mean()
    for where in ("letter", "04.results.md"):
        check(where, f"mean REBA: eldercare {reba['e']:.2f}, logistics {reba['l']:.2f}, manufacturing {reba['m']:.2f}",
              f"{reba['e']:.2f}/{reba['l']:.2f}/{reba['m']:.2f}")
    t = metrics("exp010")
    lo, hi = t["worker_clustered_bootstrap"]["ci95"]
    check("letter", f"[{lo:.3f}, {hi:.3f}]", f"[{lo:.3f}, {hi:.3f}]")
    check("04.results.md", f"[{lo:.3f}, {hi:.3f}]", "TOST clustered CI")
    check("04.results.md", f"is {t['mean_diff_eldercare_minus_manufacturing']:.3f} RULA", "TOST mean difference")
    n_m = t["n_workers"]["manufacturing"]
    check("04.results.md", f"({n_m} manufacturing-rollup workers, {t['n_workers']['eldercare']} eldercare caregivers)", "clusters")
    check("04.results.md", f"widens {t['ci_width_inflation_clustered_over_naive']:.2f}-fold", "CI widening")

# --- process ranking (Section 4.3, Fig. 7) --------------------------------------
section("ranking")
xw = pd.read_csv(BATCH / "summary_with_xw.csv")
g = xw.groupby("process_label").reba_grand.agg(["mean", "size"])
top = g[g["size"] >= 30].sort_values("mean", ascending=False)["mean"].head(8).round(2).tolist()
for k_, v in enumerate(top):
    check("04.results.md", f"(REBA {v:.2f};" if k_ == 0 else f"({v:.2f})", f"top-8 process mean REBA {v:.2f}")

n15 = int((g["size"] >= 30).sum())
check("04.results.md", f"The {W[n15]} ground-truth processes with n of at least 30, ranked by mean REBA grand score", "Fig. 7 caption")

# --- calibration (Section 4.5, Table 2) -------------------------------------------
if section("calibration", has_metrics("exp005", "exp012")):
    c = pd.DataFrame(metrics("exp005")["per_row"])
    for m in ("rula", "reba"):
        x = c[c.method == m]
        b, a = x.ece_before.mean(), x.ece_after.mean()
        check("04.results.md", f"by {100*(1-a/b):.0f}% on average ({b:.3f}" if m == "rula" else f"by {100*(1-a/b):.0f}% ({b:.3f}",
              f"{m} {b:.3f}->{a:.3f}")
        check("04.results.md", f"({b:.3f} to {a:.3f})", f"{b:.3f} to {a:.3f}")
    for _, r in c.iterrows():
        row = f"| {r.domain.capitalize()} | {r.method.upper()} | {float(r.temperature)} | {r.ece_before:.3f} | {r.ece_after:.3f} |"
        check("04.results.md", row, "Table 2 row")
    t_opt = c.temperature.unique()
    check("04.results.md", f"ECE at T_opt ({t_opt[0]}, the grid's lower bound)", "Table 6 caption: T_opt", len(t_opt) == 1)
    check("04.results.md", f"Dashed: T = {t_opt[0]}, fitted for self-consistency", "Fig. 8 caption: T_opt", len(t_opt) == 1)
    w = metrics("exp012")
    for m in ("owas", "rula", "reba"):
        x = w["manufacturing_worker_held_out_repeated"][f"manufacturing/{m}"]
        lo, hi = x["ece_after_min_max"]
        check("04.results.md", f"{x['ece_after_mean']:.3f} ({lo:.3f}-{hi:.3f})", f"held-out manufacturing {m}")
        for dom in ("eldercare", "logistics"):
            x = w[f"{dom}_leave_one_worker_out"]["summary"][f"{dom}/{m}"]
            lo, hi = x["ece_after_min_max"]
            check("04.results.md", f"{x['ece_after_mean']:.3f} ({lo:.3f}-{hi:.3f})", f"leave-one-worker-out {dom} {m}")
    out = run("mc_coverage")
    cov = {m: re.search(rf"{m}: 5-95% interval covers ([\d.]+)% of all poses, ([\d.]+)% .*mode matches ([\d.]+)%", out).groups()
           for m in ("OWAS", "RULA", "REBA")}
    cov = {m: [float(v) for v in x] for m, x in cov.items()}
    ow = "100" if cov["OWAS"][0] >= 99.95 else f"{cov['OWAS'][0]:.1f}"
    check("04.results.md", f"contains it for {cov['RULA'][0]:.0f}% and {cov['REBA'][0]:.0f}% ({ow}% for OWAS)", "MC coverage all (results)")
    check("04.results.md", f"the modal score matches it for {cov['RULA'][2]:.0f}% and {cov['REBA'][2]:.0f}% of RULA and REBA poses ({cov['OWAS'][2]:.1f}% for OWAS)", "mode match")
    # under-confidence at T = 1: mean top-class accuracy minus mean top-class confidence per domain (Fig. 8)
    mcs = pd.read_csv(BATCH / "summary_with_mc.csv").set_index("file_stem")
    zz = np.load(BATCH / "mc_distributions.npz", allow_pickle=True)
    dd = mcs.loc[zz["file_stem"], "factory_id"].map({6: "e", 7: "l"}).fillna("m").to_numpy()
    gap = {}
    for m_, col_ in (("owas", "owas_action"), ("rula", "rula_grand"), ("reba", "reba_grand")):
        P = zz[f"{m_}_probs"]
        keep = P.sum(1) >= 1e-6
        P = P / np.where(keep, P.sum(1), 1)[:, None]
        hit = P.argmax(1) + 1 == mcs.loc[zz["file_stem"], col_].astype(int).to_numpy()
        gap[m_] = [hit[keep & (dd == k_)].mean() - P.max(1)[keep & (dd == k_)].mean() for k_ in "elm"]
    rr_ = gap["rula"] + gap["reba"]
    check("04.results.md", f"exceeds mean top-class confidence by {min(rr_):.2f} to {max(rr_):.2f} across domains", "T=1 under-confidence RULA/REBA")
    check("04.results.md", f"under-confident, by {min(gap['owas']):.2f} to {max(gap['owas']):.2f}", "T=1 under-confidence OWAS")
    for line in run("supp_calibration_alternatives").splitlines():
        if line.startswith("|"):
            check("09.supplementary.md", line.strip(), "Table S2.1 row")
    for line in run("supp_temperature_grid").splitlines():
        p = [x.strip() for x in line.strip("|").split("|")]
        if len(p) == 7 and p[1] in ("RULA", "REBA"):
            check("09.supplementary.md", f"| {p[0].capitalize()} | {p[1]} | {p[2]} | {p[3]} | {p[4]} | {p[5]} |", "Table S2.2 row")

# --- biomechanics (Section 4.6) ---------------------------------------------------
section("biomechanics")
d = pd.read_csv(BATCH / "analysis/cohort_disc_perpose.csv")
xw_dom = xw.factory_id.map({6: "e", 7: "l"}).fillna("m")
e_m, m_m = xw.biomech_L5S1_Nm[xw_dom == "e"], xw.biomech_L5S1_Nm[xw_dom == "m"]
check("04.results.md", f"{e_m.median():.1f}", f"eldercare L5/S1 median {e_m.median():.1f} Nm")
check("04.results.md", f"{m_m.median():.1f}", f"manufacturing L5/S1 median {m_m.median():.1f} Nm")
for thr, label in [(3400, "AL"), (6400, "MPL")]:
    k = int((d.C_L5S1 >= thr).sum())
    check("04.results.md", f"{100*k/len(d):.1f}%", f"{label} {100*k/len(d):.1f}%")
    check("04.results.md", wilson(k, len(d)), f"{label} CI {wilson(k, len(d))} (Wilson)")
med = d.groupby("domain").C_L5S1.median()
for dname in ["eldercare", "manufacturing", "logistics"]:
    check("04.results.md", f"{med[dname]:,.0f}", f"{dname} disc median {med[dname]:,.0f} N")
H, p = kruskal(*[x.C_L5S1 for _, x in d.groupby("domain")])
check("04.results.md", f"H = {H:.1f}", f"H={H:.1f} p={p:.1e} eps2={H/(len(d)-1):.4f}")
check("04.results.md", f"{H/(len(d)-1):.4f}", "disc eps2")
sp = pd.read_csv(BATCH / "analysis/dense_spine.csv")
for dname, x in sp.groupby("domain"):
    k = int((x.flex_L3_deg > 30).sum())
    check("04.results.md", f"{100*k/len(x):.1f}%", f"{dname} L3>30 {100*k/len(x):.1f}%")
    check("04.results.md", wilson(k, len(x)), f"{dname} L3>30 CI {wilson(k, len(x))} (Wilson)")
r = pd.read_csv(BATCH / "analysis/cohort_disc_risk_by_profile.csv")
r = r[r.scope == "ALL"].set_index("profile")
for prof, col in [("25 M", "surgical"), ("65 F", "surgical"), ("25 M", "safe"), ("65 F", "safe")]:
    v = r.loc[prof, col] / 100
    lo, hi = proportion_confint(round(v * 6007), 6007, method="wilson")
    check("04.results.md", f"[{100*lo:.1f}, {100*hi:.1f}]", f"{prof} {col} {100*v:.1f} [{100*lo:.1f}, {100*hi:.1f}]")
for prof, x in r.iterrows():
    check("04.results.md", f"| {prof} | {x.safe:.1f} | {x.caution:.1f} | {x.elevated:.1f} | {x.surgical:.1f} |", "Table 3 row")
lo_p, hi_p = r.loc["25 M", "surgical"], r.loc["65 F", "surgical"]
if lo_p > 0:
    check("06.conclusion.md", f"roughly {round(hi_p / lo_p)}-fold spread", f"{hi_p / lo_p:.1f}x")
    check("04.results.md", f"a roughly {round(hi_p / lo_p)}-fold increase", f"{hi_p / lo_p:.1f}x")
else:
    check("06.conclusion.md", f"moves from none to {hi_p:.1f}%", "25 M exceeds in no pose")
arm = {a: [float(x) for x in re.search(rf"moment arm {a} m: AL-exceed ([\d.]+)%  MPL-exceed ([\d.]+)%",
                                         run("fig_moment_arm_sensitivity")).groups()] for a in ("0.04", "0.05", "0.06")}
check("04.results.md", f"from {arm['0.05'][0]:.1f}% to {arm['0.04'][0]:.1f}% (at 0.04 m) or {arm['0.06'][0]:.1f}% (at 0.06 m)", "moment arm AL")
check("04.results.md", f"from {arm['0.05'][1]:.1f}% to {arm['0.04'][1]:.1f}% or {arm['0.06'][1]:.1f}%", "moment arm MPL")
out = run("cohort_correlations")
rho = float(re.search(r"REBA grand: Spearman rho = ([\d.]+)", out).group(1))
check("04.results.md", f"REBA grand score = {rho:.2f}", "L5/S1-REBA rho")
rr = re.search(r"ratio = ([\d.]+), bootstrap 95% CI \[([\d.]+), ([\d.]+)\], one-sided Mann-Whitney p = ([\d.e+-]+)", out)
check("04.results.md", f"ratio of medians {float(rr.group(1)):.2f}, bootstrap 95% CI [{rr.group(2)}, {rr.group(3)}]; one-sided Mann-Whitney p = {rr.group(4)}", "L5/S1 eldercare/manufacturing")

# --- Shapley (Section 4.7) --------------------------------------------------------
if section("shapley", has_metrics("exp001c")):
    sh = metrics("exp001c")
    check("04.results.md", f"{sh['monotonicity_random_checks_rula']:,}", "RULA random checks")
    check("04.results.md", f"{sh['monotonicity_random_checks_reba']:,}", "REBA random checks")
    inc = json.loads((has_metrics("exp001c")[0].parent / "raw.json").read_text())["owas_wilson_ci_random"]
    w = (f"{100*inc['rate']:.2f}% of the randomly drawn one-step increments ({inc['inc_violations']} of {inc['inc_checks']:,}; "
         f"Wilson 95% CI [{100*inc['wilson_95_lo']:.2f}, {100*inc['wilson_95_hi']:.2f}]%), "
         f"{100*sh['monotonicity_random_pooled_owas']:.2f}% when increments and decrements are pooled "
         f"({round(sh['monotonicity_random_pooled_owas']*sh['monotonicity_random_checks_owas'])} of {sh['monotonicity_random_checks_owas']:,}), "
         f"and {100*sh['monotonicity_owas_exhaustive_inc_rate']:.2f}% on the exhaustive grid")
    check("04.results.md", "violates monotonicity in " + w, f"OWAS violation {w}")
    check("04.results.md", f"{100*sh['monotonicity_owas_exhaustive_inc_rate']:.2f}%", "OWAS exhaustive rate")
    n_trials = round(853 / sh["T16_efficiency_exact_zero_fraction_owas"])  # zero fraction = k/1010
    check("04.results.md", f"{n_trials:,} verification trials", f"{n_trials} trials")
    eff = max(sh[f"T16_efficiency_max_residual_{m}"] for m in ("owas", "rula", "reba"))
    add = max(sh[f"additivity_independent_max_dev_{m}"] for m in ("owas", "rula", "reba"))
    check("04.results.md", "residual below 9e-15", f"{eff:.1e}", eff < 9e-15)
    check("04.results.md", "below 2e-14", f"{add:.1e}", add < 2e-14)
    log = run("fig_shapley_attribution")
    for axis, fmt in (("Neck", "neck accounts for {:.0f}%"), ("Upper arm", "upper arm {:.0f}%"), ("Trunk", "trunk {:.0f}%"), ("Lower arm", "lower arm {:.0f}%")):
        v = float(re.search(rf"{axis}\s+([\d.]+)%", log).group(1))
        check("04.results.md", fmt.format(v), f"{axis} {v}%")
    tn = float(re.search(r"trunk \+ neck = ([\d.]+)%", log).group(1))
    check("04.results.md", f"trunk and neck together {tn:.0f}%", "trunk+neck")
    g, n_top = re.search(r"n poses at RULA grand (\d+): (\d+)", log).groups()
    check("04.results.md", f"Shapley attribution for the {n_top} poses at the highest RULA grand score in the cohort", "Fig. 10 caption")
    check("04.results.md", f"all {n_top} poses at the highest RULA grand score in the cohort ({g})", "highest-RULA poses")

# --- guarded fusion (Section 4.4, Fig. 5) -------------------------------------------
FUS = E / "result/exp-r2-fusion-comparison-v2/summary_table.csv"
if section("fusion", [BATCH / "analysis/v1_vs_v2_clinical_summary.csv", BATCH / "analysis/v1_vs_v2_clinical.csv", FUS]):
    n_poses = pd.read_csv(BATCH / "analysis/v1_vs_v2_clinical.csv").groupby("joint_name").file_stem.nunique()
    n_meth = pd.read_csv(FUS).query("condition == 'same_person'").method.pipe(lambda m: m[~m.str.contains("_minus_")]).nunique()
    check("04.results.md", f"Section 4.4 compares the {W[n_meth]} fusion methods quantitatively", "fusion methods compared", n_poses.nunique() == 1)
    check("04.results.md", f"Fig. 5 shows on {W[n_poses.max()]} illustrative poses", "Fig. 5 poses")
    check("04.results.md", f"on the {W[n_poses.max()]}-pose-per-joint panel of Section 4.4", "Fig. 5 text: panel")
    check("04.results.md", f"(illustrative panel of {W[n_poses.max()]} high-risk poses per joint)", "Fig. 5 caption")
    v = pd.read_csv(BATCH / "analysis/v1_vs_v2_clinical_summary.csv").set_index("joint_name")
    a = lambda j, c: f"{v.loc[j, c]:.1f}"
    check("04.results.md", f"({a('L5/S1','v1_mean')} to {a('L5/S1','v2-CD-soft_mean')} Nm)", "soft L5/S1")
    check("04.results.md", f"({a('Knee','v1_mean')} to {a('Knee','v2-CD-soft_mean')} Nm and {a('Hip','v1_mean')} to {a('Hip','v2-CD-soft_mean')} Nm)", "soft knee, hip")
    check("04.results.md", f"({a('Shoulder','v1_mean')} to {a('Shoulder','v2-CD-soft_mean')} Nm)", "soft shoulder")
    check("04.results.md", f"(hip {a('Hip','v1_mean')} to {a('Hip','v2-CD-max_mean')} Nm, shoulder {a('Shoulder','v1_mean')} to "
          f"{a('Shoulder','v2-CD-max_mean')} Nm, knee {a('Knee','v1_mean')} to {a('Knee','v2-CD-max_mean')} Nm", "WTA hip, shoulder, knee")
    g = v["v2-guarded_delta"].abs().max()
    check("04.results.md", "guarded fusion reproduces the single-backbone values", f"max |guarded delta| {g}", g == 0)

# --- backbone confidence (Supplementary Table S3.1) ------------------------------------
mac = BATCH / "analysis/model_accuracy_comparison.csv"
if section("backbones", [mac]):
    bk = pd.read_csv(mac)
    fm = lambda v: "100" if round(v, 1) == 100 else f"{v:.1f}"
    names = {"Body (COCO-17)": "RTMPose-Body", "BodyWithFeet (Halpe-26)": "RTMPose-BodyWithFeet",
             "Wholebody3d → BIOERA-25": "RTMW3D-x, major-25"}
    n_w = bk.groupby("model").size()
    check("04.results.md", f"of the {W[len(n_w)]} rtmlib backbones on the {W[n_w.max()]} example workers", "S3 pointer", n_w.nunique() == 1)
    for m_, g_ in bk.groupby("model"):
        check("09.supplementary.md", f"| {names[m_]} | {int(g_.n_kpts.iloc[0])} | {g_.pct_visible.mean():.1f} "
              f"({fm(g_.pct_visible.min())}-{fm(g_.pct_visible.max())}) | {g_.mean_conf.mean():.2f} |", "Table S3.1 row")

# --- cross-method agreement (Table 4) ------------------------------------------------
section("agreement")
for a_, b_, kl, kq, rho in re.findall(r"(owas|rula)\s+(rula|reba)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+6007",
                                     run("fig_cross_method_agreement").replace("         reba", "owas     reba")):
    check("04.results.md", f"| {a_.upper()} vs {b_.upper()} | {float(kl):.3f} | {float(kq):.3f} | {float(rho):.3f} |", "Table 4 row")
ag = pd.read_csv(BATCH / "analysis/cross_method_agreement.csv").set_index(["method_a", "method_b"])
check("letter", f"quadratic-weighted kappa {ag.loc[('rula', 'reba'), 'kappa_quadratic']:.2f} between RULA and REBA", "kappa (letter)")
check("letter", f"{ag.loc[('owas', 'rula'), 'kappa_quadratic']:.2f} and {ag.loc[('owas', 'reba'), 'kappa_quadratic']:.2f} "
      "between OWAS and RULA or REBA", "kappa (letter)")
check("letter", f"rank-correlate at rho {ag.loc[('rula', 'reba'), 'rho']:.2f}", "rho (letter)")
check("letter", f"rank correlations ({ag.loc[('owas', 'rula'), 'rho']:.2f} and {ag.loc[('owas', 'reba'), 'rho']:.2f}",
      "rho (letter)")

# --- context defaults, contact rule, classifier (Sections 3.9, 4.8, 4.9) ---------
section("context")
# the script prints one decimal; a printed x.5 is a tie (31.5 is 31.48 = 1,891 of 6,007), so round it down
m = re.search(r"conservative \(both\)\s+mean diff \+([\d.]+)\s+level 1->2 ([\d.]+)%\s+any upward level shift ([\d.]+)%", run("ctx_default_shift"))
check("04.results.md", f"raises the mean REBA grand score by {float(m.group(1)):.2f} points across the 6,007 poses and moves {float(m.group(3)):.0f}% of poses up at "
      f"least one action level ({float(m.group(2)) - 1e-6:.0f}% from level 1 to 2)", "context default shift")
check("letter", f"REBA plus {float(m.group(1)):.2f} points", "context default shift (letter)")
out = run("cohort_correlations")
check("09.supplementary.md", f"Spearman rho = {float(re.search(r'S8:.*rho = ([\d.]+)', out).group(1)):.2f}", "S8 rho")
from sklearn.metrics import balanced_accuracy_score, f1_score
pr = pd.read_csv(BATCH / "summary_with_pred.csv").dropna(subset=["pred_process_id"])
y, yh = pr.process_id, pr.pred_process_id.astype(int)
check("04.results.md", f"reaches {100*(y == yh).mean():.1f}% top-1 accuracy, macro-F1 {f1_score(y, yh, average='macro'):.3f}, "
      f"and balanced accuracy {balanced_accuracy_score(y, yh):.3f} across {y.nunique()} process classes (n = {len(pr):,}", "classifier")
pr["pp"] = yh
hot = {f_: x.groupby("pp").rula_grand.agg(["mean", "size"]).sort_values("mean").iloc[-1] for f_, x in pr[pr.factory_id <= 5].groupby("factory_id")}
plab = pr.drop_duplicates("process_id").set_index("process_id").process_label
hl = {f_: plab[int(r.name)] for f_, r in hot.items()}
sites = lambda pred: "(" + ", ".join(f"F{f_}" for f_ in hl if pred(hl[f_])) + ")"
num = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
n_sas = sum(v.startswith("사상") for v in hl.values())
sas = " or ".join(sorted({"Sasang-" + v[2:] for v in hl.values() if v.startswith("사상")}))
small = [f_ for f_ in hot if hot[f_]["size"] <= 2]
check("04.results.md", f"a predicted Sasang process ({sas}) the hottest cell at {num[n_sas]} of the five manufacturing sites {sites(lambda v: v.startswith('사상'))}, "
      f"a welding process at {sites(lambda v: v.startswith('용접'))[1:-1]}, and a predicted caregiving class (bathing), which does not occur at that site, at "
      f"{sites(lambda v: v.startswith('목욕'))[1:-1]}; at {num[len(small)]} of these sites ({', '.join(f'F{f_}' for f_ in small)}) the hottest cell holds only one or two poses",
      str({f_: (hl[f_], int(hot[f_]['size'])) for f_ in hl}))
pt = json.loads((BATCH / "process_training.json").read_text())
check("04.results.md", f"held-out test split it reaches {100*pt['test_accuracy']:.1f}% and macro-F1 {pt['test_macro_f1']:.3f}", "classifier test split")
cache = ROOT / "ace_pose/_runs/contact_cohort/cache"
n_cache = len(list(cache.glob("*.npz"))) if cache.is_dir() else 0
if n_cache < len(s):
    section("contact")
    skipped.append(("contact", f"second-person cache has {n_cache} of {len(s)} images"))
elif section("contact", [cache]):
    out = run("contact_cohort_score")
    rate = {d_: float(x) for d_, x in re.findall(r"^(\w+)\s+n=\s*\d+.*contact fires\s+\d+ \(([\d.]+)%\)", out, re.M)}
    check("04.results.md", f"fires on {rate['eldercare']:.1f}% of eldercare, {rate['logistics']:.1f}% of logistics, and {rate['manufacturing']:.1f}% of manufacturing poses",
          rate)

# --- VLM context comparison (Section 4.8) -------------------------------------------
vlm = ROOT / "bioera/vlm_comparison/result/exp-vlm50-context-comparison/per_image.csv"
if section("vlm", [vlm]):
    v = pd.read_csv(vlm)
    k = int((v.v_load_band == "unknown").sum())
    check("04.results.md", f"({k} of 50 recorded as `unknown`)", f"{k} unknown load_band")
    flag = v[v.h_load_band != "unknown"].v_load_band.value_counts()
    check("04.results.md", f"of the {len(v[v.h_load_band != 'unknown'])} poses", str(flag.to_dict()))
    fl = v[v.h_load_band != "unknown"]
    agree = int((fl.v_load_band == fl.h_load_band).sum())
    check("04.results.md", f"agreed on {agree} of {len(fl)}", str(flag.to_dict()))
    pct = {f: round(100 * (v[f"h_{f}"].astype(str) == v[f"v_{f}"].astype(str)).mean()) for f in
           ("static", "coupling", "repetitive", "rapid", "load_band")}
    norm = round(100 * (v.h_load_band.replace("unknown", "none") == v.v_load_band).mean())
    check("04.results.md", f"activity-static ({pct['static']}%) and moderate on coupling ({pct['coupling']}%)", pct)
    check("04.results.md", f"activity-repetitive ({pct['repetitive']}%) and activity-rapid ({pct['rapid']}%)", pct)
    check("04.results.md", f"agreement is {pct['load_band']}% only", "raw load")
    check("04.results.md", f"semantically normalised, agreement is {norm}%", "normalised load")
    check("letter", f"{pct['static']} percent on the activity-static flag, {pct['coupling']} percent on coupling, "
          f"{pct['repetitive']} percent on activity-repetitive, and {pct['rapid']} percent on activity-rapid", pct)
    check("letter", f"Raw string-match agreement was {pct['load_band']} percent", "raw load (letter)")
    check("letter", f"agreement is {norm} percent", "normalised load (letter)")
    check("letter", f"of the {len(fl)} poses", "flagged load (letter)")

# --- expert comparison (Section 4.9, letter); the expert's scores are not released -----
exp_csv = ROOT / "ace_pose/_runs/expert_pilot/expert_vs_pipeline.csv"
exp_scores = ROOT / "datasets/public/expert_pilot_22/expert_pilot_scores.csv"
if section("expert", [exp_csv, exp_scores, E / "result/exp-r2-expert-agreement/agreement.csv", E / "result/exp-r2-expert-agreement/per_image.csv"]):
    x = pd.read_csv(exp_csv)
    em = x.groupby("standard").exact_match.sum()
    check("letter", f"{em['OWAS']}/22 for OWAS, {em['RULA']}/22 for RULA, and {em['REBA']}/22 for REBA", em.to_dict())
    under = int((x[x.standard == "RULA"].direction == "under").sum())
    check("letter", f"{under} of 22 images", f"RULA under {under}")
    dis = x[~x.exact_match.astype(bool)]
    for kw in ("load", "walk", "repet"):
        n = int(dis.mechanisms.fillna("").str.contains(kw, case=False).sum())
        check("letter", f"{n} of {len(dis)}", f"{kw} {n}/{len(dis)}")
    ps = pd.read_csv(exp_scores)
    for std, before in [("OWAS", "from 3 to"), ("REBA", "from 5 to")]:
        pv = ps[ps.standard == std].pivot(index="id", columns="rater", values="legs")
        n = int((pv.system_submitted == pv.expert).sum())
        check("letter", before, f"{std} legs before fix {n}/22", f"from {n} to" == before)
    lf = pd.read_csv(ROOT / "ace_pose/_runs/expert_pilot/load_fix_before_after.csv")
    # Section 4.9 correction 2: leg sub-score agreement with the expert, submitted vs current code
    import contextlib
    import io
    import fig_expert_load_fix as xf
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        xf.leg_subscore_check(ps)
    lg = {k: (int(a), int(b)) for k, a, b in re.findall(r"(OWAS|REBA) legs: submitted (\d+)/22, current code (\d+)/22", buf.getvalue())}
    check("04.results.md", f"rose from {lg['OWAS'][0]} to {lg['OWAS'][1]} of 22 for the OWAS legs code and from {lg['REBA'][0]} to {lg['REBA'][1]} of 22 for the REBA leg score", lg)
    # Table 11: three settings by three standards, from the registered agreement table
    ag11 = pd.read_csv(E / "result/exp-r2-expert-agreement/agreement.csv")
    pi11 = pd.read_csv(E / "result/exp-r2-expert-agreement/per_image.csv")
    for stg, label in (("submitted", "Submitted"), ("cohort", "Cohort"), ("taskload", "Task-load")):
        for std in ("OWAS", "RULA", "REBA"):
            a = ag11[(ag11.setting == stg) & (ag11.method == std)].iloc[0]
            base = pi11[(pi11.setting == stg) & (pi11.method == std)].expert.value_counts().max()
            check("04.results.md", f"| {label} | {std} | {a.exact_n} | {base} | {100 * a.within1_rate:.1f} | "
                  f"{a.kappa_linear:.3f} ({a.kappa_linear_lo:.3f} to {a.kappa_linear_hi:.3f}) |", f"Table 11 {stg} {std}")
    cs = {m_: ag11[(ag11.setting == "cohort") & (ag11.method == m_)].iloc[0].exact_n for m_ in ("OWAS", "RULA", "REBA")}
    cb = {m_: pi11[(pi11.setting == "cohort") & (pi11.method == m_)].expert.value_counts().max() for m_ in ("OWAS", "RULA", "REBA")}
    check("04.results.md", f"exact agreement is {cs['OWAS']}, {cs['RULA']}, and {cs['REBA']} of 22 for OWAS, RULA, and REBA", "cohort exact (Table 11 text)")
    check("04.results.md", f"would agree on {cb['OWAS']}, {cb['RULA']}, and {cb['REBA']} images", "baseline (Table 11 text)")
    fin = {s: int((g.taskload == g.expert).sum()) for s, g in lf.groupby("standard")}
    pos = {s: int((g.posture == g.expert).sum()) for s, g in lf.groupby("standard")}
    check("letter", f"exact agreement with the expert is {fin['OWAS']}, {fin['RULA']}, and {fin['REBA']} of 22 for OWAS, RULA, and REBA",
          "pilot, final code (letter)")
    check("letter", f"The posture corrections alone give {pos['OWAS']}, {pos['RULA']}, and {pos['REBA']}",
          "pilot, posture corrections (letter)")

    # Section 4.9 text: sample, disagreement mechanisms, sub-score agreement, load sensitivity
    nimg = x.id.nunique()
    dm = x.drop_duplicates("id").domain.value_counts()
    check("04.results.md", f"cohort images ({dm['eldercare']} eldercare, {dm['logistics']} logistics, {dm['manufacturing']} manufacturing;", "pilot sample", nimg == 22)
    check("00.abstract.md", f"No evidence of expert agreement on {nimg} development images (one rater)", "highlight 1")
    check("04.results.md", f"expert action level (columns), n = {nimg} per panel", "Fig. 13 caption")
    mech = dis.mechanisms.fillna("")
    cnt = lambda kw: int(mech.str.contains(kw, case=False).sum())
    check("04.results.md", f"reasons for the {len(dis)} disagreeing image-standard pairs", "disagreeing pairs")
    check("04.results.md", f"**Load**: {cnt('load')} of the {len(dis)} reasons cite a load", "mechanism load")
    check("04.results.md", f"**Legs/locomotion**: {cnt('walk')} reasons cite walking", "mechanism walking")
    check("04.results.md", f"**Sustained or repeated activity**: {cnt('repet')} reasons cite a held static posture", "mechanism repetition")
    check("04.results.md", f"{int((mech == '').sum())} of the {len(dis)} cite none", "mechanism none")
    sub = lambda std, col: int((lambda pv: (pv.system_submitted == pv.expert).sum())(ps[ps.standard == std].pivot(index="id", columns="rater", values=col)))
    check("04.results.md", f"RULA upper arm {sub('RULA', 'upper_arm')}/22, lower arm {sub('RULA', 'lower_arm')}/22, neck {sub('RULA', 'neck')}/22, trunk {sub('RULA', 'trunk')}/22; "
          f"REBA trunk {sub('REBA', 'trunk')}/22, neck {sub('REBA', 'neck')}/22, legs {sub('REBA', 'legs')}/22; OWAS legs {sub('OWAS', 'legs')}/22", "sub-score agreement")
    check("04.results.md", f"Of the {22 - sub('OWAS', 'legs')} OWAS legs disagreements", "OWAS legs disagreements")
    check("04.results.md", f"RULA wrist agrees on {sub('RULA', 'wrist')}/22", "RULA wrist agreement")
    tl = {m_: ag11[(ag11.setting == "taskload") & (ag11.method == m_)].iloc[0].exact_n for m_ in ("OWAS", "RULA", "REBA")}
    check("04.results.md", f"exact agreement is {tl['RULA']} of 22 for RULA instead of {cs['RULA']} and {tl['REBA']} for REBA instead of {cs['REBA']}, and OWAS stays at {tl['OWAS']}",
          "load sensitivity", tl["OWAS"] == cs["OWAS"])

# --- Section 4.10: annotation benchmark, annotated-keypoint pose vs expert, measurable share ---
AA = ROOT / "bioera/analysis/result/action_agreement"
if section("limits", [AA / "metrics.json", AA / "metrics_split.json"]):
    am, sp = json.loads((AA / "metrics.json").read_text()), json.loads((AA / "metrics_split.json").read_text())
    acc = {k: sp[f"test20_{k}"] for k in ("OWAS", "RULA", "REBA")}
    for k, name in (("OWAS", "OWAS"), ("RULA", "RULA"), ("REBA", "REBA")):
        a, ci = acc[k]["accuracy"], acc[k]["ci95"]["accuracy"]
        check("04.results.md", f"{100 * a:.1f}% for {name}" + (f" (95% bootstrap interval {100 * ci[0]:.1f}-{100 * ci[1]:.1f})"
              if k == "OWAS" else f" ({100 * ci[0]:.1f}-{100 * ci[1]:.1f})"), f"test20 accuracy {k}")
    check("04.results.md", "kappa " + ", ".join(f"{acc[k]['kappa_quadratic']:.2f}" for k in ("OWAS", "RULA"))
          + f", and {acc['REBA']['kappa_quadratic']:.2f}", "test20 quadratic kappa")
    cov = am["coverage"]
    check("04.results.md", f"{100 * cov['view30']['OWAS']['all']:.1f}% of the 5,889 annotated images are fully measurable for OWAS"
          f" and {100 * cov['view30']['RULA']['all']:.1f}% for RULA and REBA", "measurable share 30 deg")
    la = lambda k, v: f"{100 * am[f'labels_{v}_{k}']['accuracy']:.1f}"
    check("04.results.md", f"rises from {la('OWAS', 'all')}% (all 5,889 annotated images) to {la('OWAS', 'view30')}% for OWAS"
          f" and from {la('RULA', 'all')}% to {la('RULA', 'view30')}% for RULA, while REBA does not rise ({la('REBA', 'all')}% to {la('REBA', 'view30')}%)",
          "measurable subset agreement")
# needs the expert's scores, which are not released: its own section, so that a
# release without them skips it instead of shifting the numbering of the checks above
if section("limits_expert", [ROOT / "datasets/public/expert_pilot_22/expert_pilot_scores.csv", AA / "metrics.json"]):
    am = json.loads((AA / "metrics.json").read_text())
    o = {k: round(am[f"expert22_annotated_pose_{k}"]["accuracy"] * am[f"expert22_annotated_pose_{k}"]["n"])
         for k in ("OWAS", "RULA", "REBA")}
    check("04.results.md", f"gives exact agreement with the expert of {o['OWAS']}, {o['RULA']}, and {o['REBA']} of 22",
          "annotated-keypoint pose vs expert")

# --- Supplementary S9: trunk-flexion error against motion capture (3DPW test/validation, CarDA) ---
S9 = {"3DPW test": ROOT / "bioera/analysis/result/threedpw_angles/summary.json",
      "3DPW validation": ROOT / "bioera/analysis/result/threedpw_angles/validation/summary.json",
      "CarDA": ROOT / "bioera/analysis/result/carda_angles/summary.json"}
if section("supp_s9", list(S9.values())):
    for name, f in S9.items():
        mae = json.loads(f.read_text())["all"]["mae"]
        if "bal3d" in mae:
            cells = [mae[m]["trunk_flexion"] for m in ("planar", "free3d", "rom3d", "bal3d", "prior")]
            check("09.supplementary.md", f"| {name} | " + " | ".join(f"{v:.1f}" for v in cells) + " |", f"Table S9.1 {name}")

# --- paragraph coverage: other-pose-model comparison (Section 4.9), highlights ----------------
PM = E / "result/pose_model_comparison"
if section("pose_models", [PM / "accuracy_speed.csv", PM / "lift3d_timing.csv"]):
    pa, pt_ = pd.read_csv(PM / "accuracy_speed.csv"), pd.read_csv(PM / "lift3d_timing.csv")
    alt = pa[pa.model != "rtmw3d"]
    ex = alt.expert_exact_OWAS_RULA_REBA.str.split("/", expand=True).astype(int)
    check("04.results.md", f"{W[len(alt)].capitalize()} alternative models", "alternative models")
    check("04.results.md", f"Keypoint accuracy ranged from {alt.PCK05.min():.2f} to {alt.PCK05.max():.2f} (PCK at 0.05", "PCK range")
    check("04.results.md", f"exact agreement with the expert from {ex.min().min()} to {ex.max().max()} of 22 for every standard", "exact agreement range")
    check("04.results.md", f"from {alt.gpu_ms.min():.0f} ms to {alt.gpu_ms.max():.0f} ms on a GPU, while the scoring layer takes {pt_.scoring_ms.max():.1f} ms per pose",
          "latency", pt_.scoring_ms.nunique() == 1)
AI = E / "result/exp-r2-assumed-inputs/metrics.json"
IC = E / "result/exp-r2-indep-calibration/metrics.json"
if section("highlights", [AI, IC, AA / "metrics.json"]):
    am_ = json.loads(AI.read_text())
    wr = lambda std, k: am_[f"wrist_worst/{std}/ALL/{k}"]
    check("00.abstract.md", f"Worst-case wrist moved RULA/REBA action level in {100 * wr('RULA', 'share_action_changed'):.1f}%/"
          f"{100 * wr('REBA', 'share_action_changed'):.1f}% of {wr('RULA', 'n'):,} images", "highlight: worst wrist")
    ic = json.loads(IC.read_text())
    cells = [k for k in ic if k.endswith("/pass_def_indep")]
    k_pass = sum(bool(ic[k]) for k in cells)
    check("00.abstract.md", f"calibration met its criterion in only {k_pass} of {len(cells)} cells", "highlight: calibration cells", k_pass == ic["criterion/cells_pass_def"])
    cv = json.loads((AA / "metrics.json").read_text())["coverage"]["view30"]
    check("00.abstract.md", f"Only about {round(100 * cv['OWAS']['all'])}% of workplace images are fully measurable under our visibility rule", "highlight: measurable share",
          abs(cv["RULA"]["all"] - cv["OWAS"]["all"]) < 0.05)  # "about 30%" must also fit RULA/REBA (31.5%)
if section("highlights_s9", list(S9.values())):
    mae = [json.loads(f.read_text())["all"]["mae"]["planar"]["trunk_flexion"] for f in S9.values()]
    check("00.abstract.md", f"Image-plane trunk angle errs {round(min(mae))}-{round(max(mae))} degrees against public motion capture", "highlight: trunk error")

# --- code-correction history (Section 4.9) ---------------------------------------------
# Cohort summaries kept from before each correction. They are not in the review copy, so this section is skipped there.
pre_table, pre_legs = BATCH / "_backup_pre_tablefix_20261004/summary.csv", BATCH / "_backup_pre_legsfix_20261005/summary.csv"
pre_planar = BATCH / "_backup_pre_planar_20261008/analysis/cohort_disc_perpose.csv"
if section("history", [pre_table, pre_legs, pre_planar, exp_scores]):
    from ace_pose.ergo_v2.tables import RULA_TABLE_A, RULA_TABLE_C  # noqa: E402
    a = pd.read_csv(pre_table)                                                  # stale Table A
    b = pd.read_csv(pre_legs).set_index("file_stem").loc[a.file_stem].reset_index()  # re-scored; leg heuristic still in place
    level = lambda g: np.digitize(g, [3, 5, 7]) + 1                             # RULA grand 1-2, 3-4, 5-6, 7 -> action level 1-4
    d = b.rula_action.values - a.rula_action.values
    check("04.results.md", f"Re-scoring changed {(d != 0).sum()} of the {len(a):,} RULA action levels ({100 * (d != 0).mean():.1f}%), {(d > 0).sum()} of them upward",
          "stale run vs re-scored run", (level(a.rula_grand) == a.rula_action).all())
    # the table correction alone: current Table A on the stale run's own arm and wrist scores (wrist twist 1, as that run scored it)
    sa = RULA_TABLE_A[a.rula_upper_arm - 1, a.rula_lower_arm - 1, a.rula_wrist - 1, 0]
    t = level(RULA_TABLE_C[sa - 1, a.rula_score_b - 1]) - a.rula_action.values
    check("04.results.md", f"changed {(t != 0).sum()}, all upward", "current Table A on the stale sub-scores", (t >= 0).all())
    one = int(b.owas_legs.isin([3, 5]).sum())                                   # OWAS legs codes 3 and 5: weight on one leg
    check("04.results.md", f"gave a one-legged legs code to {one:,} of the {len(b):,} poses ({100 * one / len(b):.0f}%)", "legs codes 3 and 5 before the fix")
    check("04.results.md", f"an implausible {pd.read_csv(pre_planar).C_L5S1.max() / 1000:.0f} kN in one pose", "largest L5/S1 compression before the fix")
    x = pd.read_csv(exp_scores).query("standard == 'RULA' and rater == 'expert'")
    hit = int((RULA_TABLE_A[x.upper_arm.astype(int) - 1, x.lower_arm.astype(int) - 1, x.wrist.astype(int) - 1, 0] == x.score_a).sum())
    check("04.results.md", f"reproduces the expert's own Table-A lookups in {hit} of {len(x)} rows", "current Table A on the expert's arm and wrist scores")

# --- unit tests -----------------------------------------------------------------------
section("tests")
ignore = [f"--ignore=ace_pose/tests/{t}" for t in LEGACY_TESTS if (ROOT / "ace_pose/tests" / t).exists()]
out = subprocess.run([PY, "-m", "pytest", "-q", "ace_pose/tests", *ignore], cwd=ROOT, capture_output=True, text=True).stdout
n_tests = re.search(r"(\d+) passed", out)
n_fail = re.search(r"(\d+) failed", out)
check("03.methods.md", f"ships {n_tests.group(1) if n_tests else '?'} unit tests", out.strip().splitlines()[-1] if out.strip() else "no output", not n_fail)

# --- report ---------------------------------------------------------------------------
w2 = min(70, max(len(r["needle"]) for r in rows))
for r in rows:
    print(f"{r['verdict']:8s} {r['section']:11s} {r['where']:19s} {r['needle'][:w2]:{w2}s} {r['value']}")
for name, why in skipped:
    print(f"SKIP     {name:11s} {why}")
if not MANUSCRIPT_MODE:
    missing = set(EXP) - {(r["section"], r["i"]) for r in rows} - {k for k in EXP if k[0] in dict(skipped)}
    for k in sorted(missing):
        print(f"MISSING  {k[0]:11s} expected check #{k[1]} was not produced: {EXP[k]['needle']}")
    rows.extend({"verdict": "MISSING"} for _ in missing)
bad = sum(r["verdict"] != "MATCH" for r in rows)
print(f"\n{len(rows) - bad} MATCH, {bad} not matching, {len(skipped)} section(s) skipped"
      f" ({'manuscript' if MANUSCRIPT_MODE else 'release'} mode)")
if MANUSCRIPT_MODE and "--write-expected" in sys.argv:
    if bad:
        sys.exit("not writing expected values while checks fail")
    EXPECTED.write_text(json.dumps([{k: r[k] for k in ("section", "i", "where", "needle", "value")} for r in rows], indent=1))
    print(f"wrote {EXPECTED.relative_to(ROOT)}")
sys.exit(1 if bad else 0)
