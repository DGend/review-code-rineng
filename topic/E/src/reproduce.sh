#!/usr/bin/env bash
# PWW-DISCLOSURE: public
# Regenerate every result of the paper from the per-image pose caches, then check
# the numbers. Run from anywhere:  bash topic/E/src/reproduce.sh [--images] [--quick]
#
#   (default)  re-score the 6,007 cached poses and re-run every analysis that
#              does not need the images (about 30-60 min on a laptop CPU).
#   --images   first re-run pose inference and the image-based steps; needs the
#              AI-Hub images and labels (set BIOERA_DATA_ROOT, see README).
#   --quick    skip the cohort re-scoring chain; only re-run the analyses and checks.
#
# Results go to topic/E/result/reproduce/, logs to topic/E/result/reproduce/logs/;
# the script stops at the first failure.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
B=ace_pose/_runs/batch_v1
IMAGES=0; QUICK=0
for a in "$@"; do
  case "$a" in --images) IMAGES=1 ;; --quick) QUICK=1 ;; *) echo "unknown option $a"; exit 2 ;; esac
done
OUT=topic/E/result/reproduce
mkdir -p "$OUT/logs" topic/E/paper/figures/supplementary
step() {  # step <name> <command...>
  local name="$1"; shift
  printf '%-34s' "$name"
  local t0=$SECONDS
  if "$@" > "$OUT/logs/$name.log" 2>&1; then echo "ok  ($((SECONDS - t0)) s)"
  else echo "FAILED, see $OUT/logs/$name.log"; tail -n 20 "$OUT/logs/$name.log"; exit 1; fi
}

if [ "$IMAGES" = 1 ]; then
  step pose_inference           "$PY" -m ace_pose.batch_infer --out "$B"
fi
if [ "$QUICK" = 0 ]; then
  rm -f "$B/analysis/dense_spine.csv"   # cache; force recomputation
  step rescore_cohort           "$PY" ace_pose/rescore_batch.py --batch "$B"
  step process_classifier       "$PY" ace_pose/train_process_classifier.py --batch "$B" --device cpu
  step monte_carlo              "$PY" ace_pose/run_c3_mc.py --batch "$B"
  step temperature_calibration  "$PY" ace_pose/run_c4_calibration.py --batch "$B"
  step biomechanics             "$PY" ace_pose/run_x_w_batch.py --batch "$B"
  step cohort_analysis          "$PY" ace_pose/run_analysis.py --batch "$B"
  step cohort_plots             "$PY" ace_pose/run_plots.py --batch "$B"
  step disc_tolerance           "$PY" ace_pose/run_cohort_biomech.py --batch "$B"
  step dense_spine              "$PY" ace_pose/run_dense_spine.py --batch "$B"
fi
if [ "$IMAGES" = 1 ]; then
  step joint_focus              "$PY" ace_pose/run_joint_focus.py --batch "$B"
  step guarded_fusion_panel     "$PY" ace_pose/run_v1_vs_v2_clinical.py --batch "$B"
  step backbone_confidence      "$PY" ace_pose/run_model_accuracy_comparison.py --batch "$B"
  step backbone_overlays        "$PY" ace_pose/run_pose_overlay_comparison.py --batch "$B"
  step convention_grid          "$PY" ace_pose/run_full_model_comparison.py --batch "$B"
  step pose_overview            "$PY" ace_pose/run_pose_visualization.py --batch "$B"
  step load_overview            "$PY" ace_pose/run_env_load_visualization.py --batch "$B"
  step second_person_inference  "$PY" topic/E/src/contact_cohort_infer.py
  if [ -f datasets/public/expert_pilot_22/expert_pilot_scores.csv ]; then
    step expert_load_fix        "$PY" topic/E/src/fig_expert_load_fix.py
  fi
fi
step second_person_contact      "$PY" topic/E/src/contact_cohort_score.py
for e in exp001c_shapley_hardening:exp001c exp002_standards_audit:exp002 exp005_per_domain_ece:exp005 \
         exp006_load_chain_r:exp006 exp010_worker_cluster_tost:exp010 exp011_cd4_worker_cluster_ece:exp011 \
         exp012_c4_worker_holdout_refit:exp012; do
  step "${e#*:}" env PWW_RESULT_DIR="$OUT/${e#*:}" "$PY" "topic/A/src/public/${e%%:*}.py"
done
for f in fig_process_ranking fig_shapley_attribution fig_cross_method_agreement fig_demographic_disc fig_guarded_fusion; do
  step "$f" "$PY" "topic/E/src/$f.py"
done
if [ -f datasets/public/expert_pilot_22/expert_pilot_scores.csv ]; then
  step fig_expert_comparison    "$PY" topic/E/src/fig_expert_comparison.py
fi
# action-level agreement: expert pilot (if released) and dataset-annotation reference (Section 4.9);
# without the AI-Hub annotations it recomputes the metrics from the shipped per-image levels
step action_agreement           "$PY" topic/E/src/eval_action_agreement.py
step action_agreement_split     "$PY" topic/E/src/eval_action_agreement_split.py
step collect_figures            "$PY" topic/E/src/collect_figures.py
step paper_numbers              "$PY" topic/E/src/verify_manuscript_numbers.py
tail -n 3 "$OUT/logs/paper_numbers.log"
