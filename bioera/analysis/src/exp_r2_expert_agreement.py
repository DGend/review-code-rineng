# PWW-DISCLOSURE: public
"""exp-r2-expert-agreement: 전문가 22장과 시스템 행동 수준의 표준 일치 지표(정확 일치, 한 단계 이내,
선형/이차 가중 kappa, 부호 있는 평균 차이, 이미지 단위 부트스트랩 95% 구간).

설계(paper/targets/targets.md 마지막 절): 두 설정으로 22장의 시스템 점수를 다시 계산한다.
  cohort   (가) 코호트 본분석 설정: cohort_context(하중 0 kg, 손목 기본값 그대로). exp_r2_assumed_inputs 의 baseline 과 같은 호출.
  taskload (나) 전문가 비교 설정: task_context(작업별 잠정 하중표) = 원고 표 10 의 8단계.
  submitted    전문가에게 제출됐던 시스템 점수(등록 자료 그대로, 재계산 아님) = 표 10 의 첫 행.
자세는 코호트 캐시(resolve("cohort_pose_cache"))의 22장 기존 추론을 쓴다(새 추론 없음).
표 10 의 1~7단계는 접촉 하중 규칙(이미지 재추론으로 얻는 제2 인물 검출 필요)과 폐기된 이전 각도 코드를
쓰므로 여기서 재현하지 않는다.

Outputs ($PWW_RESULT_DIR): metrics.json, agreement.csv, per_image.csv.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pww_data import resolve  # noqa: E402
from ace_pose.ergo_v2.context_policy import cohort_context, task_context  # noqa: E402
import exp_r2_assumed_inputs as ai  # noqa: E402  (load_poses/run 재사용)

OUT = Path(os.environ.get("PWW_RESULT_DIR", ROOT / "bioera/analysis/result/exp-r2-expert-agreement"))
METHODS = ("OWAS", "RULA", "REBA")
# 방법별 가능한 행동 수준 전체(관측값으로 만들지 않는다): reba.py:181-189, rula.py:189, owas 1-4
LEVELS = {"OWAS": [1, 2, 3, 4], "RULA": [1, 2, 3, 4], "REBA": [0, 1, 2, 3, 4]}
ACTION_COL = {"OWAS": "owas_action", "RULA": "rula_action", "REBA": "reba_action"}
# 원고 표 10 의 기존 정확 일치 장수(04.results.md Table 10): 첫 행과 8단계
MANUSCRIPT_EXACT = {("submitted", "OWAS"): 12, ("submitted", "RULA"): 5, ("submitted", "REBA"): 8,
                    ("taskload", "OWAS"): 12, ("taskload", "RULA"): 10, ("taskload", "REBA"): 10}
MANUSCRIPT_MAD = {("submitted", "OWAS"): 0.86, ("submitted", "RULA"): 1.27, ("submitted", "REBA"): 0.82,
                  ("taskload", "OWAS"): 0.77, ("taskload", "RULA"): 0.73, ("taskload", "REBA"): 0.68}
N_BOOT, SEED = 2000, 0


def weighted_kappa(a, b, levels, power):
    """가중 kappa (power 1 선형, 2 이차). 범주 집합은 levels 로 고정. 정의 불가면 NaN."""
    k = len(levels)
    idx = {l: i for i, l in enumerate(levels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    W = (np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)) ** power
    den = (W * E).sum()
    return float("nan") if den == 0 else float(1 - (W * O).sum() / den)


def stats(s, e, levels):
    """시스템 s, 전문가 e 행동 수준 배열에 대한 지표 사전."""
    d = s - e
    return {"exact_rate": float((d == 0).mean()), "within1_rate": float((np.abs(d) <= 1).mean()),
            "kappa_linear": weighted_kappa(s, e, levels, 1), "kappa_quadratic": weighted_kappa(s, e, levels, 2),
            "mean_signed_diff": float(d.mean())}


def bootstrap(s, e, levels, n_boot=N_BOOT, seed=SEED):
    """이미지 단위 부트스트랩 백분위 95% 구간. 표본에서 kappa 가 정의되지 않으면 그 표본은 해당 지표에서 뺀다."""
    rng = np.random.default_rng(seed)
    draws = [stats(s[i], e[i], levels) for i in (rng.integers(0, len(s), len(s)) for _ in range(n_boot))]
    out = {}
    for key in draws[0]:
        v = np.array([d[key] for d in draws])
        ok = v[~np.isnan(v)]
        out[key] = (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))) if len(ok) else (float("nan"),) * 2
        out[key + "_undefined_draws"] = int(np.isnan(v).sum())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(resolve("expert_pilot_22"))
    ids = scores.drop_duplicates("id")[["id", "file_stem", "domain"]].reset_index(drop=True)
    summ = pd.read_csv(Path(ai.BATCH) / "summary.csv")
    sub = ids.merge(summ[summ.status == "ok"], on=["file_stem"], how="left", suffixes=("", "_c"))
    assert sub.process_label.notna().all(), "22장이 코호트 캐시에 모두 있어야 한다"
    poses = ai.load_poses(sub)

    system = {"cohort": ai.run(poses, cohort_context), "taskload": ai.run(poses, task_context)}
    levels_of = lambda st, m, r: scores[(scores.standard == m) & (scores.rater == r)].set_index("id").action.reindex(ids.id).to_numpy().astype(int)
    system["submitted"] = pd.DataFrame({ACTION_COL[m]: levels_of(None, m, "system_submitted") for m in METHODS})

    metrics, rows, per = {"n_images": len(ids)}, [], []
    for setting, df in system.items():
        for m in METHODS:
            s, e = df[ACTION_COL[m]].to_numpy().astype(int), levels_of(None, m, "expert")
            lv = LEVELS[m]
            assert set(s) | set(e) <= set(lv), (setting, m)
            st, bs = stats(s, e, lv), bootstrap(s, e, lv)
            k = f"{setting}/{m}"
            metrics[f"{k}/n"] = len(s)
            metrics[f"{k}/exact_n"] = int((s == e).sum())
            row = {"setting": setting, "method": m, "n": len(s), "exact_n": int((s == e).sum())}
            for key, v in st.items():
                if np.isnan(v):
                    metrics[f"{k}/{key}_undefined"] = 1   # 한 범주만 관측 등으로 정의되지 않음: 키를 만들지 않는다
                    continue
                lo, hi = bs[key]
                metrics[f"{k}/{key}"] = v
                row[key] = v
                if not np.isnan(lo):
                    metrics[f"{k}/{key}_lo"], metrics[f"{k}/{key}_hi"] = lo, hi
                    row[key + "_lo"], row[key + "_hi"] = lo, hi
                if bs[key + "_undefined_draws"]:
                    metrics[f"{k}/{key}_boot_undefined_draws"] = bs[key + "_undefined_draws"]
            if (setting, m) in MANUSCRIPT_EXACT:
                me, mm = MANUSCRIPT_EXACT[(setting, m)], MANUSCRIPT_MAD[(setting, m)]
                mad = float(np.abs(s - e).mean())
                metrics[f"{k}/manuscript_exact_n"] = me
                metrics[f"{k}/same_exact_as_manuscript"] = int(me == row["exact_n"])
                metrics[f"{k}/mean_abs_diff"] = mad
                metrics[f"{k}/manuscript_mean_abs_diff"] = mm
                metrics[f"{k}/same_mean_abs_diff_as_manuscript"] = int(round(mad, 2) == mm)
            rows.append(row)
            per += [{"id": i, "setting": setting, "method": m, "system": int(a), "expert": int(b)}
                    for i, a, b in zip(ids.id, s, e)]
    pd.DataFrame(rows).to_csv(OUT / "agreement.csv", index=False)
    pd.DataFrame(per).to_csv(OUT / "per_image.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True))
    print(pd.DataFrame(rows)[["setting", "method", "exact_n", "kappa_linear", "kappa_linear_lo", "kappa_linear_hi"]].to_string())


if __name__ == "__main__":
    main()
