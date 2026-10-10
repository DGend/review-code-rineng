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

Usage: python topic/E/src/verify_manuscript_numbers.py [--write-expected]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parent
E = ROOT / "topic/E"
BATCH = ROOT / "ace_pose/_runs/batch_v1"
MS = E / "paper/manuscript"
LETTER = E / "paper/build/BIOERA_RiE_response_to_reviewers.md"
EXPECTED = SRC / "paper_numbers_expected.json"
REPRO = E / "result/reproduce"
PY = sys.executable
# Test files of the retired v1 scorer, not part of the released package.
LEGACY_TESTS = ["test_pipeline.py", "test_end_to_end.py", "test_gold_lifter.py"]
# reproduce.sh writes topic/E/result/reproduce/<key>/metrics.json; the dated folders are
# the archived runs the manuscript was checked against.
ARCHIVE = {
    "exp005": "topic/E/result/exp-005-per-domain-ece-mcfix-20261005",
    "exp010": "topic/E/result/exp-010-worker-cluster-tost-mcfix-20261005",
    "exp011": "topic/E/result/exp-011-worker-cluster-ece-deterministic-20261006",
    "exp012": "topic/E/result/exp-012-worker-holdout-refit-mcfix-20261005",
    "exp001c": "topic/E/result/exp-001c-shapley-hardening-legsfix-20261005",
    "exp006": "topic/E/result/reproduce/exp006",
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
check("09.supplementary.md", "Images were collected for 16", str(s.process_id.nunique()), s.process_id.nunique() == 16)

# --- standards cross-check (Section 4.2, Table 1) ------------------------------
section("standards")
sys.path.insert(0, str(ROOT / "topic/A/src/public"))
import exp002_standards_audit as audit  # noqa: E402
ow = audit.diff_owas()
total = sum(f()["total_cells"] for f in (audit.diff_owas, audit.diff_rula_A, audit.diff_rula_B, audit.diff_rula_C,
                                         audit.diff_reba_A, audit.diff_reba_B, audit.diff_reba_C))
check("03.methods.md", f"Of {total} cells", "cells audited")
check("03.methods.md", f"{ow['wrong_cells']} of {ow['total_cells']} OWAS Action-Category cells", "OWAS cells differing")
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
for v in top:
    check("04.results.md", f"{v:.2f})", f"top-8 process mean REBA {v:.2f}")

# --- calibration (Section 4.5, Table 2) -------------------------------------------
if section("calibration", has_metrics("exp005", "exp011", "exp012")):
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
    w = metrics("exp012")
    for m in ("owas", "rula", "reba"):
        x = w["manufacturing_worker_held_out_repeated"][f"manufacturing/{m}"]
        lo, hi = x["ece_after_min_max"]
        check("04.results.md", f"{x['ece_after_mean']:.3f} ({lo:.3f}-{hi:.3f})", f"held-out manufacturing {m}")
        for dom in ("eldercare", "logistics"):
            x = w[f"{dom}_leave_one_worker_out"]["summary"][f"{dom}/{m}"]
            lo, hi = x["ece_after_min_max"]
            check("04.results.md", f"{x['ece_after_mean']:.3f} ({lo:.3f}-{hi:.3f})", f"leave-one-worker-out {dom} {m}")
    k = metrics("exp011")
    f = [k[f"manufacturing/{m}"]["ci_width_inflation_clustered_over_naive"] for m in ("owas", "rula", "reba")]
    check("04.results.md", f"({f[0]:.2f}-fold for OWAS, {f[1]:.2f}-fold for RULA, {f[2]:.2f}-fold for REBA)", "CI widening")
    out = run("mc_coverage")
    cov = {m: re.search(rf"{m}: 5-95% interval covers ([\d.]+)% of all poses, ([\d.]+)% .*mode matches ([\d.]+)%", out).groups()
           for m in ("OWAS", "RULA", "REBA")}
    cov = {m: [float(v) for v in x] for m, x in cov.items()}
    check("03.methods.md", f"{cov['RULA'][1]:.0f}% (RULA), {cov['REBA'][1]:.0f}% (REBA), and {cov['OWAS'][1]:.0f}% (OWAS)", "MC coverage conf>=0.6")
    ow = "100" if cov["OWAS"][0] >= 99.95 else f"{cov['OWAS'][0]:.1f}"
    check("03.methods.md", f"({cov['RULA'][0]:.0f}%, {cov['REBA'][0]:.0f}%, and {ow}% of all poses)", "MC coverage all")
    check("04.results.md", f"contains it for {cov['RULA'][0]:.0f}% and {cov['REBA'][0]:.0f}% ({ow}% for OWAS)", "MC coverage all (results)")
    check("03.methods.md", f"matches the deterministic score for {cov['RULA'][2]:.0f}% and {cov['REBA'][2]:.0f}% of poses ({cov['OWAS'][2]:.0f}% for OWAS)", "mode match")
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
    check("06.conclusion.md", f"from none of the poses to {hi_p:.1f}%", "25 M exceeds in no pose")
arm = {a: [float(x) for x in re.search(rf"moment arm {a} m: AL-exceed ([\d.]+)%  MPL-exceed ([\d.]+)%",
                                         run("fig_moment_arm_sensitivity")).groups()] for a in ("0.04", "0.05", "0.06")}
check("04.results.md", f"from {arm['0.05'][0]:.1f}% to {arm['0.04'][0]:.1f}% (at 0.04 m) or {arm['0.06'][0]:.1f}% (at 0.06 m)", "moment arm AL")
check("04.results.md", f"from {arm['0.05'][1]:.1f}% to {arm['0.04'][1]:.1f}% or {arm['0.06'][1]:.1f}%", "moment arm MPL")
r6 = metrics("exp006")["pearson_r_overall"]
check("03.methods.md", f"Pearson r = {r6:.3f} over all 6,007 poses", "dense-spine vs joint-moment L5/S1")
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
    w = f"{100*sh['monotonicity_random_pooled_owas']:.2f}% (Wilson 95% CI [{100*sh['owas_wilson_95_lo']:.2f}, {100*sh['owas_wilson_95_hi']:.2f}]"
    check("04.results.md", w, f"OWAS violation {w}")
    check("04.results.md", f"{100*sh['monotonicity_owas_exhaustive_inc_rate']:.2f}%", "OWAS exhaustive rate")
    n_trials = round(853 / sh["T16_efficiency_exact_zero_fraction_owas"])  # zero fraction = k/1010
    check("04.results.md", f"{n_trials:,} verification trials", f"{n_trials} trials")
    eff = max(sh[f"T16_efficiency_max_residual_{m}"] for m in ("owas", "rula", "reba"))
    add = max(sh[f"additivity_independent_max_dev_{m}"] for m in ("owas", "rula", "reba"))
    check("04.results.md", "residual below 9e-15", f"{eff:.1e}", eff < 9e-15)
    check("04.results.md", "below 2e-14", f"{add:.1e}", add < 2e-14)
    log = run("fig_shapley_attribution")
    for axis, fmt in (("Neck", "neck accounts for {:.0f}%"), ("Upper arm", "upper arm {:.0f}%"), ("Trunk", "trunk {:.0f}%")):
        v = float(re.search(rf"{axis}\s+([\d.]+)%", log).group(1))
        check("04.results.md", fmt.format(v), f"{axis} {v}%")
    tn = float(re.search(r"trunk \+ neck = ([\d.]+)%", log).group(1))
    check("04.results.md", f"trunk and neck together {tn:.0f}%", "trunk+neck")
    g, n_top = re.search(r"n poses at RULA grand (\d+): (\d+)", log).groups()
    check("04.results.md", f"all {n_top} poses at the highest RULA grand score in the cohort ({g})", "highest-RULA poses")

# --- guarded fusion (Section 4.4, Fig. 5) -------------------------------------------
if section("fusion", [BATCH / "analysis/v1_vs_v2_clinical_summary.csv"]):
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
m = re.search(r"conservative \(both\)\s+mean diff \+([\d.]+)", run("ctx_default_shift"))
check("03.methods.md", f"under-reporting REBA by {float(m.group(1)):.1f} points", "context default shift")
check("letter", f"REBA plus {float(m.group(1)):.2f} points", "context default shift (letter)")
out = run("cohort_correlations")
check("09.supplementary.md", f"Spearman rho = {float(re.search(r'S8:.*rho = ([\d.]+)', out).group(1)):.2f}", "S8 rho")
from sklearn.metrics import balanced_accuracy_score, f1_score
pr = pd.read_csv(BATCH / "summary_with_pred.csv").dropna(subset=["pred_process_id"])
y, yh = pr.process_id, pr.pred_process_id.astype(int)
check("04.results.md", f"reaches {100*(y == yh).mean():.1f}% top-1 accuracy, macro-F1 {f1_score(y, yh, average='macro'):.3f}, "
      f"and balanced accuracy {balanced_accuracy_score(y, yh):.3f} across {y.nunique()} process classes (n = {len(pr):,}", "classifier")
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
    check("04.results.md", f"{rate['eldercare']:.1f}% of eldercare poses", "contact eldercare")
    check("04.results.md", f"{rate['logistics']:.1f}% of logistics", "contact logistics")
    check("04.results.md", f"{rate['manufacturing']:.1f}% of manufacturing", "contact manufacturing")
    ch = {m_: int(n) for m_, n in re.findall(r"(RULA|REBA): (\d+) of 1006 poses changed", out)}
    check("04.results.md", f"raising {round(100*ch['RULA']/1006)}% of eldercare RULA and {round(100*ch['REBA']/1006)}% of eldercare REBA", "contact changes")

# --- VLM context comparison (Section 4.8) -------------------------------------------
vlm = ROOT / "topic/D/result/exp-vlm50-context-comparison/per_image.csv"
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

# --- expert pilot (Section 4.9, letter); needs the expert's consent to release -----
exp_csv = ROOT / "ace_pose/_runs/expert_pilot/expert_vs_pipeline.csv"
exp_scores = ROOT / "datasets/public/expert_pilot_22/expert_pilot_scores.csv"
if section("expert", [exp_csv, exp_scores]):
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
    ex = {s: int((g.angles_nocontact == g.expert).sum()) for s, g in lf.groupby("standard")}
    r = lf[lf.standard == "RULA"]
    under = int((r.angles_nocontact < r.expert).sum())
    # Table 5 (code-correction stages): every row from the stage file
    for col in ("submitted", "tables", "legs", "contact", "worker", "angles", "posture", "wrist", "taskload"):
        g3 = [lf[lf.standard == s] for s in ("OWAS", "RULA", "REBA")]
        cells = [str(int((g[col] == g.expert).sum())) for g in g3] + [f"{(g[col] - g.expert).abs().mean():.2f}" for g in g3]
        check("04.results.md", " | " + " | ".join(cells) + " |", f"Table 5 stage {col}")
    fin = {s: int((g.taskload == g.expert).sum()) for s, g in lf.groupby("standard")}
    pos = {s: int((g.posture == g.expert).sum()) for s, g in lf.groupby("standard")}
    check("04.results.md", f"agreement at {pos['OWAS']}, {pos['RULA']}, and {pos['REBA']} of 22 for OWAS, RULA, and REBA",
          "stage 6 agreement")
    check("04.results.md", f"agreement is {fin['OWAS']}, {fin['RULA']}, and {fin['REBA']} of 22 (Fig. 13, bottom row)",
          "stage 8 agreement")
    check("04.results.md", f"agreement is {ex['OWAS']}, {ex['REBA']}, and {ex['RULA']} of 22 for OWAS, REBA, and RULA, "
          f"with RULA below the expert in {under} of 22", "pilot, cohort code (no contact rule)")
    check("letter", f"exact agreement with the expert is {fin['OWAS']}, {fin['RULA']}, and {fin['REBA']} of 22 for OWAS, RULA, and REBA",
          "pilot, final code (letter)")
    check("letter", f"The posture corrections alone give {pos['OWAS']}, {pos['RULA']}, and {pos['REBA']}",
          "pilot, posture corrections (letter)")

# --- Section 4.10: annotation benchmark, annotated-keypoint pose vs expert, measurable share ---
AA = ROOT / "topic/E/result/action_agreement"
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
    check("04.results.md", f"from {100 * am['labels_all_OWAS']['accuracy']:.1f}% to {100 * am['labels_view30_OWAS']['accuracy']:.1f}% for OWAS"
          f" and from {100 * am['labels_all_RULA']['accuracy']:.1f}% to {100 * am['labels_view30_RULA']['accuracy']:.1f}% for RULA",
          "measurable subset agreement")
# needs the expert's scores, which ship only once the expert consents: its own section, so that a
# release without them skips it instead of shifting the numbering of the checks above
if section("limits_expert", [ROOT / "datasets/public/expert_pilot_22/expert_pilot_scores.csv", AA / "metrics.json"]):
    am = json.loads((AA / "metrics.json").read_text())
    o = {k: round(am[f"expert22_annotated_pose_{k}"]["accuracy"] * am[f"expert22_annotated_pose_{k}"]["n"])
         for k in ("OWAS", "RULA", "REBA")}
    check("04.results.md", f"gives exact agreement with the expert of {o['OWAS']}, {o['RULA']}, and {o['REBA']} of 22",
          "annotated-keypoint pose vs expert")

# --- Supplementary S9: trunk-flexion error against motion capture (3DPW test/validation, CarDA) ---
S9 = {"3DPW test": ROOT / "topic/E/result/threedpw_angles/summary.json",
      "3DPW validation": ROOT / "topic/E/result/threedpw_angles/validation/summary.json",
      "CarDA": ROOT / "topic/E/result/carda_angles/summary.json"}
if section("supp_s9", list(S9.values())):
    for name, f in S9.items():
        mae = json.loads(f.read_text())["all"]["mae"]
        if "bal3d" in mae:
            cells = [mae[m]["trunk_flexion"] for m in ("planar", "free3d", "rom3d", "bal3d", "prior")]
            check("09.supplementary.md", f"| {name} | " + " | ".join(f"{v:.1f}" for v in cells) + " |", f"Table S9.1 {name}")

# --- unit tests -----------------------------------------------------------------------
section("tests")
ignore = [f"--ignore=ace_pose/tests/{t}" for t in LEGACY_TESTS if (ROOT / "ace_pose/tests" / t).exists()]
out = subprocess.run([PY, "-m", "pytest", "-q", "ace_pose/tests", *ignore], cwd=ROOT, capture_output=True, text=True).stdout
n_tests = re.search(r"(\d+) passed", out)
n_fail = re.search(r"(\d+) failed", out)
check("03.methods.md", f"ships {n_tests.group(1) if n_tests else '?'} unit tests", out.strip().splitlines()[-1] if out.strip() else "no output", not n_fail)

# --- runtime (hardware-dependent; working tree only) ---------------------------------
# Runtime reference: the May 2026 runs (pose inference on an RTX 4090, Monte Carlo on CPU without contention).
RT = ROOT / "ace_pose/_runs/runtime_reference"
pose_log, mc_log = RT / "pose_inference_gpu_202605.log", RT / "monte_carlo_cpu_202605.log"
if MANUSCRIPT_MODE and section("runtime", [pose_log, mc_log]):
    pose_s = float(re.search(r"Done in ([\d.]+)s", pose_log.read_text()).group(1))
    mc_s = float(re.findall(r"Done in ([\d.]+)s", mc_log.read_text())[0])
    check("03.methods.md", f"pose {pose_s/60:.1f} min", f"{pose_s:.0f} s")
    check("03.methods.md", f"MC layer {mc_s/60:.1f} min", f"May 2026 c3_mc run {mc_s:.0f} s")
    check("03.methods.md", "under 30 min", f"pose+MC {(pose_s+mc_s)/60:.1f} min", (pose_s + mc_s) / 60 < 30)
elif not MANUSCRIPT_MODE:
    skipped.append(("runtime", "hardware-dependent, not re-checked in release mode"))

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
