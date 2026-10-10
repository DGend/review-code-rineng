# BIOERA: code and derived data for peer review

This repository accompanies the manuscript "BIOERA: A reproducible, calibrated, and explainable
engineering-informatics pipeline for monocular ergonomic-risk decision support across
multi-domain industrial workforces". It is an anonymised copy for review; the full repository
will be released under the MIT licence on acceptance.

Everything needed to regenerate the paper's numbers from the per-image pose caches is
included. The source images are not redistributed (see "Data" below).

## Quick start

```bash
conda env create -f environment.yml && conda activate bioera
python -m pytest -q ace_pose/tests          # unit tests
bash topic/E/src/reproduce.sh --quick       # analyses + number check, about 5 min
bash topic/E/src/reproduce.sh               # also re-scores all 6,007 poses, about 30-60 min on CPU
```

`reproduce.sh` ends with `verify_manuscript_numbers.py`, which recomputes every checked
number of the paper from the regenerated results and compares it with the value printed
in the paper (`topic/E/src/paper_numbers_expected.json` lists each value, the manuscript
section or table it appears in, and how it is computed). The last line reads
`N MATCH, 0 not matching`. Sections that need data not included here are listed as SKIP
with the reason.

## Layout

| Path | Contents |
|---|---|
| `ace_pose/ergo_v2/` | OWAS, RULA, and REBA scoring (Section 3.3, Algorithm 1) |
| `ace_pose/ergo_v2/context_policy.py`, `ace_pose/standards/task_context.json` | Task-level load for the inputs a single frame cannot show; each entry names its source and status |
| `ace_pose/ergo_v2/measurability.py` | Which scoring items an image can measure (joint visibility, side view) |
| `ace_pose/standards/` | Machine-readable OWAS/RULA/REBA specifications and the cross-check notes |
| `ace_pose/major_joints.py` | RTMW3D 133-keypoint to major-25 reduction (Section 3.2) |
| `ace_pose/batch_infer.py` | Pose inference over the cohort and selection of the labelled worker in each image |
| `ace_pose/confidence_mc.py`, `run_c3_mc.py` | Confidence-aware Monte Carlo scoring (Section 3.4, Algorithm 2) |
| `ace_pose/domain_calibration.py`, `run_c4_calibration.py` | Per-domain temperature calibration (Section 3.5) |
| `ace_pose/biomechanics.py`, `dense_spine.py`, `disc_load.py`, `demographic_load.py` | Static joint moments, dense spine, disc compression and demographic tolerance (Section 3.6) |
| `ace_pose/bioera_v2.py` | Guarded multi-backbone fusion (Section 3.7, Algorithm 3) |
| `ace_pose/xai.py` | Exact Shapley attribution (Section 3.8) |
| `ace_pose/vlm_context.py` | Heuristic and vision-language-model context extraction (Section 3.9) |
| `ace_pose/tests/` | Unit tests, including table cells pinned to the published tables |
| `topic/A/src/public/` | Standards cross-check, Shapley axioms, calibration and equivalence experiments |
| `topic/E/src/` | Analyses and figures of this manuscript, `reproduce.sh`, and the number check |
| `topic/E/src/fig_expert_load_fix.py`, `fig_expert_comparison.py` | Expert pilot: code-correction stages (Table 5) and confusion matrices (Fig. 13) |
| `topic/E/src/eval_action_agreement*.py` | Action-level agreement (confusion matrices, accuracy, F1, kappa) against the expert and against the dataset annotations, with the frozen 80/20 split |
| `topic/E/src/select_heldout_images.py` | Draws and freezes the held-out images for the second expert round |
| `topic/D/src/` | Heuristic vs vision-language-model comparison on 50 images (Section 4.8) |
| `ace_pose/_runs/batch_v1/` | Derived per-pose data (below) |

## Data

The images and labels are the Busan multi-domain HSE dataset, distributed by the Korean
AI-Hub platform (https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71439) under its own
access terms, so they are not included. This repository contains data derived from them:

- `ace_pose/_runs/batch_v1/details/<image>.npz`: the analysed worker's pose for each of the
  6,007 images (RTMW3D 3D and 2D keypoints with confidences, and the major-25 pose). Face and
  hand keypoints are set to zero; no analysis uses them.
- `ace_pose/_runs/batch_v1/summary*.csv`: per-pose metadata and scores at each stage of the
  pipeline; `mc_distributions.npz`, `c4_*`: Monte Carlo distributions and calibration;
  `analysis/*.csv`: cohort analyses.
- `ace_pose/_runs/contact_cohort/`: 2D poses of all other people detected in each image (Section 4.9).
- `topic/D/result/exp-vlm50-context-comparison/per_image.csv`: per-image outputs of the two
  context backends.
- `topic/E/result/reproduce/*/metrics.json`: the experiment outputs the paper reports.
- `topic/E/result/action_agreement/`: per-image action levels from the annotated and the estimated
  keypoints (`per_image_labels.csv`), the metrics, and the frozen 20% evaluation list with its SHA-256.
- `topic/E/result/heldout_selection/`: the 36 held-out images for the second expert round, drawn and
  frozen (SHA-256 alongside) before the corrections of Section 4.9 were made.

Steps that need the images (pose inference, the guarded-fusion panel, the backbone
comparison, figure overlays, the vision-language-model run) are run with
`reproduce.sh --images` after downloading the dataset and setting `BIOERA_DATA_ROOT` to the
folder that holds `new_images/` and `new_json/`. With the dataset's `data_v5.csv` in that folder,
`eval_action_agreement.py` recomputes the per-image levels from the annotations; without it, it
recomputes the metrics from the shipped `per_image_labels.csv`.

## Not included, and why

| Item | Reason | What the code does without it |
|---|---|---|
| Source images and annotations | Distributed by AI-Hub under its own access terms | Runs from the shipped pose caches; image steps need `--images` |
| Face and hand keypoints | Privacy; no analysis uses them | Set to zero in the pose caches |
| Expert's numeric scores for the 22 pilot images | Added to `datasets/public/expert_pilot_22/` once the expert consents to their release | Expert steps are skipped and listed as SKIP by the number check |
| Expert's free-text reasons | Not released | Only keyword-derived mechanism flags are used |

## Notes

- In each image the analysed person is the labelled worker, matched in 2D image coordinates
  (`batch_infer.py`, `_pick_primary_person`). At the eldercare and logistics sites the `W`
  token in the image file name identifies a recording session; `worker_id` in the tables is the
  labelled worker and `session_id` keeps the file-name token.
- Random steps are seeded; re-running the chain reproduces the shipped tables.
