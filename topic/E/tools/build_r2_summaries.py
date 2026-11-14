# PWW-DISCLOSURE: public
"""RiE R2 리비전 실험의 summary.md 를 결과 파일에서 다시 계산해 만든다.

심사(run_review.sh)는 summary.md 만 본다. 수치를 손으로 옮기지 않으려고 표는 전부
metrics.json 과 CSV 에서 읽어 생성하고, 사실과 사전 기준의 기계적 결과만 적는다(판정 아님).
사용: python topic/E/tools/build_r2_summaries.py   (topic/E/result/<exp>/summary.md 를 쓴다)
"""
import ast
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "result"
TH = {"OWAS": 0.05, "RULA": 0.07, "REBA": 0.10}
REF = ("참조는 사람이 표시한 2D 관절점을 같은 채점기로 채점한 값이다. 전문가 판단도 실제 인간공학 위험도 아니다.")


def md(df, nd=3):
    """DataFrame 을 마크다운 표로(수치는 nd 자리 반올림)."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values]
    return "\n".join(rows)


def run_info(exp):
    """run.json 에서 재현 명령, 소요 시간, 코드 상태를 읽는다."""
    r = json.loads((R / exp / "run.json").read_text())
    return (f"- 재현 명령(topic/E 에서 run_experiment.sh 경유): `{' '.join(r['cmd'])}`\n"
            f"- 종료 코드 {r['exit_code']}, 소요 {int(r['end']) - int(r['start'])}초, git_rev {r['git_rev'][:8]}, "
            f"git_dirty {r.get('git_dirty')}. 실행 당시 래퍼가 데이터 사용 미기록 경고를 냈다(스크립트가 등록부 조회 없이 "
            f"경로를 직접 읽었음, 이후 등록부 경유로 수정했고 수치는 동일함을 확인).")


def head(exp, claim, ev, data, note):
    return (f"---\nexp_id: {exp}\nclaim: {claim}\nevidence: {ev}\ndate: 2026-11-13\ndatasets: [{data}]\n"
            f"metrics_file: result/{exp}/metrics.json\nstatus: recorded\nnotes: {note}\n---\n# {exp} 요약 (사실 기록, 판정 아님)\n")


def ece(c, a, nb=10):
    # 구간은 (lo, hi]: src/exp_r2_empirical_noise.py 의 ece_cc 와 같은 규칙이어야 한다.
    # [lo, hi) 로 세면 신뢰도 0.5(=64/128) 표본이 옆 구간으로 가서 fold 평균이 최대 0.004 달라지고 경계 칸(제조 RULA 0.0700)의 통과 여부가 뒤집힌다.
    b = np.clip(np.ceil(c * nb - 1e-12).astype(int) - 1, 0, nb - 1)
    return float(sum((b == k).mean() * abs(a[b == k].mean() - c[b == k].mean()) for k in range(nb) if (b == k).any()))



# ---- 심사 반영(2026-11-14) 보강용 공통 도구. assumed() 와 그 의존 함수(md, run_info, head, ece)는 건드리지 않는다. ----
AUDIT = "paper/audit/code-20261114-021808.json"
DOMS = ("manufacturing", "eldercare", "logistics")
DOM_KO = {"manufacturing": "제조", "eldercare": "요양", "logistics": "물류"}
SRC = Path(__file__).resolve().parents[1] / "src"


def md2(df, nd=3):
    """md() 와 같되 문자열 열의 결측도 빈칸으로 둔다(본문에 nan 이 새지 않게)."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: "" if (not isinstance(v, str) and pd.isna(v)) else v)
    return md(df, nd)


def ci(p, lo, hi, nd=3):
    """점추정 [하한, 상한]. 구간이 없으면 점추정만."""
    if pd.isna(p):
        return ""
    return f"{p:.{nd}f}" if pd.isna(lo) or pd.isna(hi) else f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def head2(exp, claim, ev, data, note, title=None):
    return (f"---\nexp_id: {exp}\nclaim: {claim}\nevidence: {ev}\ndate: 2026-11-14\ndatasets: [{data}]\n"
            f"metrics_file: result/{exp}/metrics.json\nstatus: recorded\nnotes: {note}\n---\n# {title or exp + ' 요약'} (사실 기록, 판정 아님)\n")


def n_workers():
    """영역별 작업자 수(작업자 단위 부트스트랩 파일에서 읽음)."""
    b = pd.read_csv(R / "exp-r2-selective" / "explor_boot.csv")
    return b.groupby("domain").n_workers.max().to_dict()


def scope(wk_override=None, ref=True):
    """범위 한정 문단(요약마다 한 번)."""
    wk = n_workers()
    wtxt = ", ".join(f"{DOM_KO[d]} {wk[d]}명" for d in DOMS if d in wk)
    s = ("- 범위: 이 코호트 하나(단일 코호트)에서 작업자 단위로 나눠 본 결과이며 작업자 간 교환 가능성을 가정한 평균적 결과다. "
         f"작업자 수는 {wtxt}이다. ")
    if ref:
        s += "독립 기준은 \"사람이 표시한 2D 관절점을 같은 채점기로 채점한 점수\"이며 전문가 판단도, 실제 인간공학 위험도, 3D 오차도 아니다. "
    s += ("작업자 단위 부트스트랩 구간(2,000회, 시드 0)은 제조에만 있다. 요양과 물류는 작업자가 너무 적어(4명, 6명) 구간 추정이 불안정하므로 "
          "구간을 내지 않고 작업자별 값을 싣는다.")
    return s


def run_hist(exp):
    """2절 실행 이력 문단: 검수 통과 코드 재실행, 기존 지표 동일, git_dirty 해석, 데이터 사용 기록."""
    r = json.loads((R / exp / "run.json").read_text())
    names = [d["name"] for d in r.get("data", [])]
    if not names and (R / exp / "data-used.jsonl").exists():
        names = list(dict.fromkeys(json.loads(l)["name"] for l in (R / exp / "data-used.jsonl").read_text().splitlines() if l.strip()))
    dt = ", ".join(names) if names else "없음(이 파일이 만들어진 시점의 run.json 에 기록이 비어 있음)"
    return (f"- 실행 이력: 코드 검수({AUDIT}) 통과 코드로 재실행했고 기존 지표는 이전 실행과 전부 같았다. "
            f"run.json 의 git_dirty({r.get('git_dirty')})는 실행 뒤 결과 파일이 쓰인 상태에서 기록되어 항상 True 이므로 코드 동일성은 code_hash({r['code_hash'][:12]})로 확인한다.\n"
            f"- 데이터 사용 기록(run.json 의 data): {dt}.")


def run_info2(exp):
    """6절 재현: 명령, 종료 코드, 소요, 종료 시각, git_rev, code_hash."""
    r = json.loads((R / exp / "run.json").read_text())
    return (f"- 재현 명령(topic/E 에서 run_experiment.sh 경유): `{' '.join(r['cmd'])}`\n"
            f"- 종료 코드 {r['exit_code']}, 소요 {int(r['end']) - int(r['start'])}초, 종료 {time.strftime('%Y-%m-%d %H:%M', time.localtime(int(r['end'])))}, "
            f"git_rev {r['git_rev'][:8]}, code_hash {r['code_hash'][:12]}.")


def grid_from_src(name):
    """src 의 온도 격자 정의를 그대로 읽는다(손으로 옮기지 않는다)."""
    txt = (SRC / "exp_r2_indep_calibration.py").read_text()
    m = re.search(name + r" = np\.array\((\[.*?\])", txt)
    return ast.literal_eval(m.group(1))


# ---- 2차 재심(2026-11-14) 보강용. 이미 통과한 요약의 함수는 위 도구를 바꾸지 않는다(REF 는 fusion, selective 도 쓴다). ----
REF_ANN = ("주석 참조 점수(annotation-derived score)는 사람이 표시한 2D 관절점을 같은 채점기로 채점한 점수다. "
           "전문가 판단도 실제 인간공학 위험도도 3D 오차도 아니다.")
FOLD_COLS = ("ece_fold_mean", "brier_fold_mean")


def need_cols(df, cols, name):
    """새 열이 없으면(해당 실험을 새 코드로 재실행하기 전) 무엇을 해야 하는지 말하고 멈춘다."""
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f"오류: {name} 에 열 {miss} 가 없다. exp-r2-empirical-noise 를 새 코드로 재실행해야 이 열이 생긴다.")


def sweep_fold(pivot_stat, sw, order):
    """sigma 스윕 파일에서 fold 평균 열을 (영역, 방법) x 변형 표로."""
    return sw.pivot_table(index=["domain", "method"], columns="variant", values=pivot_stat, sort=False)[order].reset_index()


def sensitivity():
    exp = "exp-r2-mc-sensitivity-v2"
    t = pd.read_csv(R / exp / "summary_table.csv")
    rb = json.loads((R / exp / "robustness.json").read_text())
    sd = pd.read_csv(R / exp / "explor_seeds.csv")
    cur = t[t.setting == "sigma0.04_S128"].set_index(["domain", "method"]).mode_match
    t["mm_diff_pp"] = [100 * (m - cur[(d, me)]) for m, d, me in zip(t.mode_match, t.domain, t.method)]
    t["ece_ok"] = t.ece_after_default_mean <= t.thresh
    inr_set = ["sigma0.02_S128", "sigma0.04_S128", "sigma0.08_S128", "sigma0.04_S64", "sigma0.04_S256", "sigma0.04_S512"]
    inr = t[t.setting.isin(inr_set)]
    g = t.groupby("setting", sort=False).agg(ece_ok_cells=("ece_ok", "sum"), mm_within2pp=("mm_diff_pp", lambda v: int((abs(v) <= 2).sum())),
                                             min_mode_match=("mode_match", "min"), max_ece_after=("ece_after_default_mean", "max"),
                                             max_ece_after_fold_max=("ece_after_default_max", "max"), max_ece_after_leaky=("ece_after_leaky", "max"),
                                             T_min=("T_default_min", "min"), T_max=("T_default_max", "max"),
                                             T_ext_min=("T_ext_min", "min"), T_ext_max=("T_ext_max", "max")).reset_index()
    bad_a = inr[~inr.ece_ok][["setting", "domain", "method", "ece_after_default_mean", "thresh"]]
    bad_b = inr[abs(inr.mm_diff_pp) > 2][["setting", "domain", "method", "mode_match", "mm_diff_pp"]]
    bullets = lambda lst: "\n".join(f"  - {x}" for x in lst)  # noqa: E731
    # 시드 표준편차(현재 설정, 시드 0,1,2)
    sdt = sd.groupby(["domain", "method"], sort=False).agg(ece_before_mean=("ece_before", "mean"), ece_before_sd=("ece_before", "std"),
                                                         ece_after_mean=("ece_after_default_mean", "mean"), ece_after_sd=("ece_after_default_mean", "std")).reset_index()
    sdt["thresh"] = [TH[m] for m in sdt.method]
    # 처리 시간, 경계 근처 비율, 정답 보유 이미지 기준 일치율
    ex = t.groupby("setting", sort=False).agg(sigma_base=("sigma_base", "first"), S=("S", "first"), sec_per_image_mean=("explor_sec_per_image_mean", "mean"),
                                              sec_per_image_median=("explor_sec_per_image_median", "mean"), near_tie_share_mean=("explor_near_tie_share", "mean"),
                                              near_tie_share_max=("explor_near_tie_share", "max"), mode_match_n_all_min=("mode_match", "min"),
                                              mode_match_annotated_min=("explor_mode_match_annotated", "min")).reset_index()
    t["mm_n_all_minus_annotated_pp"] = 100 * (t.mode_match - t.explor_mode_match_annotated)
    gap = float(t.mm_n_all_minus_annotated_pp.abs().max())
    ex_time = ex[["setting", "sigma_base", "S", "sec_per_image_mean", "sec_per_image_median"]]
    nt_m = t.pivot_table(index="setting", columns="method", values="explor_near_tie_share", aggfunc="mean", sort=False).reset_index()
    ex_tie = ex[["setting", "sigma_base", "S", "near_tie_share_mean", "near_tie_share_max", "mode_match_n_all_min", "mode_match_annotated_min"]]
    cur_cells = t[t.setting == "sigma0.04_S128"][["domain", "method", "mode_match", "explor_mode_match_annotated", "n_all", "n_cohort", "explor_near_tie_share", "coverage", "ece_before",
                                                 "T_default_mean", "ece_after_default_mean", "ece_after_default_min", "ece_after_default_max", "T_ext_mean",
                                                 "ece_after_ext_mean", "ece_after_leaky", "thresh"]]
    nt_by_sigma = ex[ex.S == 128][["sigma_base", "near_tie_share_mean"]]
    s = head2(exp, "C5", "E3 (매개변수 민감도: 자기 일관성의 안정성)", "cohort_pose_cache 6,007장",
              "1차 실행(exp-r2-mc-sensitivity)은 이미지 단위 분할이라 판정에 쓰지 않음. 이 실험은 정확도가 아니라 자기 일관성의 안정성을 잰다",
              title=exp + " 요약: 몬테카를로 설정에 대한 자기 일관성의 안정성(정확도 아님)")
    s += f"""
## 1. 목적
이 실험이 재는 것은 **자기 일관성의 안정성**이다. 섭동 크기 sigma_base 와 표본 수 S 를 한 번에 하나씩 바꿀 때, 몬테카를로 점수 분포의 최빈값이 결정론 점수와 얼마나 같은지, 그리고 온도 조정 뒤 ECE 가 얼마나 변하는지를 본다(심사자 5-2). 참조가 파이프라인 자신의 결정론 점수이므로 **정확도도 보정 타당성도 잴 수 없다.** 정답 관절점 기준 비교는 exp-r2-empirical-noise(sigma 스윕)와 exp-r2-indep-calibration 이 다룬다. 사전 기준: targets.md "exp-r2-mc-sensitivity" 와 "검수 반영 개정: exp-r2-mc-sensitivity-v2".

## 2. 설정, 실행 이력, 범위
- 9개 설정 x 9칸(영역 3 x 방법 3) = {len(t)}행. 참조는 파이프라인 자신의 결정론 점수(자기 일관성)이다. 정답 관절점도 전문가 판단도 아니다.
- 온도 적합과 보정 후 ECE 는 작업자 단위 fold(exp-r2-indep-calibration 과 같은 fold, 학습과 검증 작업자 겹침 없음을 실행 중 단언)에서 계산했다. 1차 실행은 이미지 단위 무작위 분할이었고(코드 검수 D2, D4) 그 값은 `_leaky` 열로 남겼다. ECE 는 10구간 등폭이다.
- mode_match 와 coverage 는 전체 n_all(영역별 {sorted(t.n_all.unique().tolist())}), ECE 는 정답 관절점이 있는 n_cohort(영역별 {sorted(t.n_cohort.unique().tolist())}) 이미지에서 계산했다. 두 집합이 다르다(차이의 영향은 5절).
- 사전 격자 {{0.5 ... 5}}, 확장 격자는 0.1 부터. 판정의 시드는 0.
{run_hist(exp)}
{scope(ref=False)} 이 실험의 참조는 위 독립 기준이 아니라 파이프라인 자신의 점수다.

## 3. 사전 기준의 기계적 결과
기준: sigma 0.02~0.08(S=128)과 S>=64(sigma 0.04) 범위의 모든 설정에서 (a) 9칸 모두 보정 후 ECE(작업자 분리 평균) <= 기준(OWAS 0.05, RULA 0.07, REBA 0.10), (b) 최빈값 일치율이 현재 설정 대비 +-2 %p 이내.
- 두 조건 모두 충족하지 못했다. 사전 범위를 사후 관찰에 맞춰 줄이지 않고 미충족을 그대로 보고한다.
- (a) 범위 안 {len(inr)}칸 중 기준 초과 {len(bad_a)}칸:
{bullets(rb['a_breaks'])}
- (b) 범위 안 {len(inr)}칸 중 +-2 %p 초과 {len(bad_b)}칸(방법별 {', '.join(f'{k_} {v_}칸' for k_, v_ in bad_b.method.value_counts().items())}):
{bullets(rb['b_breaks'])}

- 사후 관찰(탐색적, 판정 아님): 두 조건(보정 후 ECE 9칸 모두 기준 이하, 최빈값 일치율 9칸 모두 +-2 %p 이내)을 9칸 모두에서 만족한 설정은 {', '.join(g[(g.ece_ok_cells == 9) & (g.mm_within2pp == 9)].setting) or '없음'} 이다. 사전 범위를 이에 맞춰 줄이지 않는다.
- 기준 ECE(OWAS 0.05, RULA 0.07, REBA 0.10)는 원고의 사전 ECE 기준을 그대로 쓴 것이다(targets.md 공통).

설정별 요약(9칸 기준):
{md2(g)}

현재 설정(sigma 0.04, S 128) 9칸:
{md2(cur_cells, 4)}

## 4. 관찰된 이상 징후
- 사전 격자에서 선택된 온도는 거의 모든 칸과 fold 에서 격자 하한 0.5 이다(위 T_min, T_max). 확장 격자에서는 더 낮은 값이 선택된다. 이 방향(T<1)이 왜 반복되는지는 검정하지 않았다. 참조가 예측 자세 자신의 점수라는 순환성 때문일 것으로 해석하나 정답 기반 참조로 확인한 것은 아니다(독립 기준에서는 반대로 상한 포화, exp-r2-indep-calibration).
- 섭동이 작을수록 자기 일관성 지표가 좋아진다(sigma 0.01 에서 최빈값 일치율과 ECE 가 가장 좋다). 참조가 예측 자세 자신의 점수이기 때문일 것으로 해석한다. 그러므로 이 표에서 가장 작은 sigma 가 "최선"이라는 읽기는 정확도 주장이 아니다.
- 현재 설정에서도 fold 최대 ECE 가 기준을 넘는 칸이 있다(평균은 기준 이하여도 fold 최대는 넘을 수 있음): {', '.join(f'{r_.domain}/{r_.method} {r_.ece_after_default_max:.3f} > {r_.thresh}' for r_ in cur_cells.itertuples() if r_.ece_after_default_max > r_.thresh) or '없음'}.

## 5. 탐색적(결과를 본 뒤 추가)
통과/미달 기준 없음. 지표 키는 explor/ 접두.

### 5.1 몬테카를로 시드 0, 1, 2 (현재 설정, 표준편차)
시드를 바꿔 보정 전 ECE 와 작업자 분리 보정 후 ECE 를 다시 계산했다. 표준편차는 시드 3개의 표본 표준편차다. ECE 표준편차의 최댓값은 보정 전 {sdt.ece_before_sd.max():.4f}, 보정 후 {sdt.ece_after_sd.max():.4f} 이다. 이는 시드 변동일 뿐이고 최빈값 일치율의 시드 변동은 저장하지 않았다(explor_seeds.csv 에 ECE 열만 있음). 따라서 +-2 %p 판정의 시드 불확실성은 평가하지 못했다.
{md2(sdt, 4)}

### 5.2 처리 시간 (같은 실행 안의 설정 간 비교용)
이미지당 몬테카를로 처리 시간이다. 설정을 순서대로 한 실행 안에서 잰 값이라 설정 사이의 상대 비교에만 쓴다. 절대 시간은 기계와 부하에 따르며 현장 지연 시간으로 해석하지 않는다. 측정하지 않은 값(다른 기계, 실시간 사용)은 쓰지 않는다. 시간은 영역과 방법에 따라 달라지지 않고 설정마다 한 값이다(9칸 평균).
{md2(ex_time, 4)}

### 5.3 경계 근처 이미지 비율과 정답 관절점 보유 이미지 기준 일치율
경계 근처 비율은 1위와 2위 수준의 표본 비율 차이가 0.1 미만인 이미지의 비율이다(코드의 near_tie). 섭동이 커질수록 경계 근처 이미지가 늘어 최빈값이 불안정해지는 경로를 보여 준다. 정답 관절점 보유 이미지 기준 일치율은 mode_match 를 n_cohort 에 해당하는 이미지로 다시 센 값이며 참조는 여전히 결정론 점수다. n_all 기준과의 차이는 최대 {gap:.2f} %p 이다(칸별 값은 설정별 표의 mode_match 와 explor_mode_match_annotated).
{md2(ex_tie, 4)}

방법별 경계 근처 비율(영역 평균):
{md2(nt_m, 4)}

- 방법별로 RULA 와 REBA 가 먼저 무너지는 이유(점수 수준이 많거나 경계에 민감하다는 가설)는 검정하지 않았다. 위 경계 근처 비율을 방법별로 나눈 값은 explor_settings.csv 에 있다.
- 정답 관절점 기준으로 sigma 를 비교한 결과는 exp-r2-empirical-noise 의 explor_sigma_sweep.csv 로 넘긴다. 이 실험은 참조가 자기 점수이므로 sigma 의 정확도 비교를 하지 않는다.

## 6. 구현자가 정한 선택(사전 기준에 없음)
확장 격자의 점 구성, 최빈값 동률 시 낮은 수준 선택, mode_match 와 ECE 의 이미지 집합 차이, 경계 근처의 정의(0.1), 처리 시간의 집계.

## 7. 재현
{run_info2(exp)}
"""
    (R / exp / "summary.md").write_text(s)


def indep():
    exp = "exp-r2-indep-calibration"
    t = pd.read_csv(R / exp / "summary_table.csv")
    bt = pd.read_csv(R / exp / "explor_boot.csv")
    pw = pd.read_csv(R / exp / "explor_per_worker.csv")
    sw_path = "result/exp-r2-empirical-noise/explor_sigma_sweep.csv"
    sw = pd.read_csv(R / "exp-r2-empirical-noise" / "explor_sigma_sweep.csv")
    need_cols(sw, FOLD_COLS, sw_path)
    sw = sw[sw.target == "grand"]
    st = pd.read_csv(R / "exp-r2-mc-sensitivity-v2" / "explor_settings.csv")
    sec = st[st.S == 128].groupby("sigma_base").explor_sec_per_image_mean.mean()
    g_def, g_expl = grid_from_src("GRID_DEF"), grid_from_src("GRID_EXPLOR")
    g_ext = [0.1, 0.2, 0.3, 0.4] + g_def           # src: GRID_EXT = [0.1, 0.2, 0.3, 0.4] + GRID_DEF
    wk = n_workers()
    t = t.copy()
    t["workers"] = t.domain.map(wk)
    # 경계 칸: 기준이 (다)의 fold 범위 안에 든다(fold 에 따라 통과와 미달이 갈릴 수 있음). 사전 기준이 아니라 사후 표시.
    t["boundary"] = np.where((t.ece_c_def_min <= t.thresh) & (t.thresh <= t.ece_c_def_max), "경계", "")
    n_def, n_ext, n_mat = int(t.pass_def.sum()), int(t.pass_ext.sum()), int((t.matched_ece_c_def_mean <= t.thresh).sum())
    n_expl, n_expl_m = int(t.explor_pass.sum()), int(t.explor_matched_pass.sum())
    n_bw = int(t.b_worse_than_a.sum())
    n_bw_m = int((t.matched_ece_b > t.matched_ece_a).sum())
    n_ub = int((t.T_def_min == max(g_def)).sum())
    n_bd = int((t.boundary == "경계").sum())
    pass_cells = ", ".join(f"{DOM_KO[r.domain]} {r.method}" + (" (경계)" if r.boundary else "") for r in t[t.pass_def].itertuples()) or "없음"
    main = t[["domain", "method", "workers", "n", "n_folds", "mode_acc", "coverage", "ece_a", "ece_b", "ece_c_def_mean", "ece_c_def_min", "ece_c_def_max",
              "brier_a", "brier_b", "brier_c_def_mean", "thresh", "pass_def", "boundary"]]
    side = t[["domain", "method", "mode_acc", "ece_a", "ece_b", "ece_c_def_mean", "pass_def", "matched_mode_acc", "matched_ece_a", "matched_ece_b",
              "matched_ece_c_def_mean", "thresh"]].copy()
    side["matched_pass"] = side.matched_ece_c_def_mean <= side.thresh
    selfc = t[["domain", "method", "self_mode_acc", "self_ece_a", "self_ece_b", "self_ece_c_def_mean"]]
    ex = t[["domain", "method", "T_def_min", "T_def_max", "explor_T_mean", "explor_T_min", "explor_T_max", "explor_frac_folds_at_upper_bound", "explor_ece_mean",
            "explor_ece_min", "explor_ece_max", "explor_pass", "explor_matched_ece_mean", "explor_matched_pass", "thresh"]]
    # 작업자 단위 부트스트랩(제조) 및 요양·물류 작업자별
    b = bt[(bt.stat == "ece") & (bt.basis == "rows") & (bt.domain == "manufacturing")]
    brow = []
    for me, g in b.groupby("method", sort=False):
        q = g.set_index("analysis")
        brow.append(dict(method=me, raw_T1=ci(q.point["indep_raw"], q.ci_lo["indep_raw"], q.ci_hi["indep_raw"]),
                         calibrated_pre_grid=ci(q.point["indep_cal_def"], q.ci_lo["indep_cal_def"], q.ci_hi["indep_cal_def"]),
                         explor_grid=ci(q.point["indep_cal_explor"], q.ci_lo["indep_cal_explor"], q.ci_hi["indep_cal_explor"]), thresh=TH[me]))
    pwt = pw[pw.domain != "manufacturing"].pivot_table(index=["domain", "method", "worker", "n_images"], columns="analysis", values="ece", sort=False).reset_index()
    pwt = pwt[["domain", "method", "worker", "n_images", "indep_raw", "indep_cal_def", "indep_cal_explor"]]
    pwt["표시"] = np.where(pwt.n_images < 30, "참고(n<30)", "")
    n_small = int((pwt[pwt.method == "OWAS"].n_images < 30).sum())
    # sigma 스윕 연결(exp-r2-empirical-noise): 3절과 같은 fold 평균 집계
    sp = sweep_fold("ece_fold_mean", sw, ["gauss0.04", "gauss0.2", "sized_gauss", "A1"])
    sp["thresh"] = [TH[m] for m in sp.method]
    n_down = int((sp["gauss0.2"] < sp["gauss0.04"]).sum())
    n_p04, n_p20 = int((sp["gauss0.04"] <= sp.thresh).sum()), int((sp["gauss0.2"] <= sp.thresh).sum())
    s = head2(exp, "C5", "E2 (보정 타당성)", "cohort_pose_cache, aihub_busan_annotations 5,889장",
              "탐색 격자(0.1~50) 결과는 이 실행의 explor/ 지표이며 판정에 쓰지 않음. 별도 실행 exp-r2-indep-calibration-explor 는 중복이라 제쳐 둠")
    s += f"""
## 1. 목적과 결론
심사자 6의 지적(몬테카를로 분포의 중심과 보정 대상이 같은 예측 자세에서 나온다)에 답하기 위해, 예측 자세와 독립인 주석 참조 점수로 보정을 평가한다. {REF_ANN} 이하 "주석 참조"라 쓴다. 사전 기준: targets.md "exp-r2-indep-calibration".

**결론: 보정 주장은 성립하지 않는다. 몬테카를로 분포는 자기 일관성(예측 자세 자신의 점수에 대한 일치)만 보장한다.** 사전 기준(작업자 분리 평균 ECE 가 기준 이하인 칸이 9칸 중 7칸 이상)에 주 분석은 {n_def}칸, matched 변형은 {n_mat}칸이어서 미달이고, 원고의 서술을 "자기 일관성"으로 낮춘다. 이 결론은 이 코호트, 이 자세 추정기, 이 섭동 형태(정규 섭동)에서의 결과이며, 다른 코호트, 카메라, 작업 조건, 자세 추정기, 섭동 형태로의 일반화는 이 실험이 보지 않았다.

**사용 지침(측정된 비교만):** 주석 참조 기준에서 이 확률을 위험 확률로 읽지 않는다. 원고의 T=0.5 는 자기 일관성에서만 최적이며, 주석 참조 기준에서는 T=0.5(나)가 T=1(가)보다 ECE 를 키운 칸이 주 분석 {n_bw} / 9칸, matched {n_bw_m} / 9칸이다(3절).

**독립 참조 점검 절차(다른 파이프라인에서 재사용할 수 있는 형태):**
1. 입력: 사람이 2D 관절점을 표시한 이미지, 작업자 식별자, 파이프라인의 몬테카를로 점수 분포(배포 입력 정의 하나와 관절 집합을 주석에 맞춘 변형 하나).
2. 출력: 영역과 방법별 칸에서 작업자를 분리해 나눈 fold 의 ECE(10구간 등폭), (가) 원래 확률, (나) 현재 설정, (다) 학습 작업자에서 맞춘 온도 조정 후 값.
3. 판정식: 칸별 fold 평균 ECE 가 기준(OWAS 0.05, RULA 0.07, REBA 0.10, 이 연구가 사전에 정한 값) 이하인 칸이 9칸 중 7칸 이상이면 "주석 참조에 대해 보정됨", 미만이면 "자기 일관성"으로 서술을 낮춘다. 코드 공개는 투고 단계에서 정한다.

## 2. 설정, 실행 이력, 범위
- 분포: 원고의 현재 몬테카를로(정규 섭동 sigma 0.04, S 128, 캐시). 주석 참조: 주석 관절점을 같은 채점기와 코호트 본분석 맥락(자세 전용)으로 채점한 종합 점수(OWAS 는 행동 범주).
- 이미지 {int(t[t.method == 'OWAS'].n.sum())}장(영역별 n 은 표). 작업자 단위 fold(제조는 작업자 묶음 80/20 반복, 요양과 물류는 작업자 한 명씩 제외), fold 수는 n_folds 열. ECE 는 10구간 등폭이다.
- (가) T=1, (나) 원고의 자기 일관성 온도 T=0.5, (다) 학습 작업자에서 주석 참조에 맞춘 온도. (다)의 ECE 는 fold 평균이다.
- 온도 격자(src/exp_r2_indep_calibration.py 에서 읽음): 사전 격자 {g_def}. 확장 격자 {g_ext}(사전 기준에 함께 기록, targets.md 에 "사전 격자와 확장 격자 0.1~5"). 탐색 격자 {g_expl}는 1차 결과에서 모든 fold 가 사전 격자와 확장 격자의 상한 T=5 에 닿은 것을 본 뒤인 2026-11-13 에 추가했다(paper/decision-log.md). 탐색 격자 결과는 이 실행의 explor/ 지표이며 판정에 쓰지 않는다.
- 별도 실행으로 만든 exp-r2-indep-calibration-explor 는 같은 스크립트의 중복 실행이라 제쳐 둔다. 탐색 결과는 이 실행의 explor/ 지표(아래 5절)를 쓴다.
{run_hist(exp)}
{scope(ref=False)}

## 3. 사전 기준의 기계적 결과
기준: (다)의 작업자 분리 평균 ECE 가 기준(OWAS 0.05, RULA 0.07, REBA 0.10) 이하인 칸이 9칸 중 7칸 이상이면 "자세 오차에 대해 보정됨", 미만이면 "자기 일관성"으로 서술을 낮춘다.

**주 분석과 matched 의 역할.** 주 분석은 파이프라인이 실제로 쓰는 입력 정의(배포 조건: 예측 자세의 전체 관절)이고, matched 는 주석 참조와 관절 집합을 맞춘 비교(주석에 없는 귀, 새끼발가락, 발뒤꿈치를 뺀 입력으로 입력 정의 차이를 제거한 진단)다. 판정은 주 분석으로 읽는다. 두 분석의 최빈값 정확도 차이(표의 mode_acc 와 matched_mode_acc)는 입력 정의의 차이와 일치하지만, 관절별로 점수 변화를 분해하지는 않았다.

- 주 분석: 기준 이하 {n_def}칸 / 9칸(확장 격자 {n_ext}칸). 통과한 칸: {pass_cells}. matched(진단): {n_mat}칸 / 9칸. 판정: 미달, 서술을 낮춘다.
- (나)가 (가)보다 ECE 를 키운 칸: 주 분석 {n_bw} / 9칸, matched {n_bw_m} / 9칸. 이것이 원고의 T=0.5 가 주석 참조 기준에서 보정을 악화시킨다는 직접 결과다.
- (다)에서 선택된 온도가 사전 격자의 상한 {max(g_def)} 인 칸: {n_ub} / 9칸(T_def_min 이 상한과 같음).
- 경계 칸(사후 표시, 사전 기준 아님): 기준이 (다)의 fold 범위(ece_c_def_min 에서 ece_c_def_max) 안에 드는 칸을 주 분석 표의 boundary 열에 "경계"로 표시했다. {n_bd} / 9칸이다. matched 표에는 fold 범위 열이 없어 표시하지 못했다.

주 분석과 matched 변형을 나란히(어느 쪽도 사전 기준에 쓰였다: pass_def 는 주 분석 판정, matched_pass 는 같은 기준을 matched 에 적용한 값):
{md2(side)}

주 분석 전체(Brier, fold 범위, 경계 표시 포함):
{md2(main)}

자기 일관성 기준(참조 = 예측 자세의 결정론 점수)의 같은 지표:
{md2(selfc)}

## 4. 관찰된 이상 징후
- (다)에서 선택된 온도가 사전 격자에서 모든 칸과 fold 에서 상한이다. 자기 일관성 기준에서는 하한 0.5 가 선택된 것과 방향이 반대다.
- 상한 포화의 원인은 이 실험 안에서 검정하지 않았다. 다만 별도 실험(exp-r2-empirical-noise)의 탐색적(결과를 본 뒤 추가한) sigma 스윕과 방향이 일치한다: sigma_base 를 0.04 에서 0.20 으로 올리면 주석 참조 ECE(fold 평균)가 {n_down} / 9칸에서 줄었고, 기준 이하 칸은 {n_p04} 칸에서 {n_p20} 칸이 된다(아래 표). 이는 섭동이 실제 오차보다 작을 가능성과 일치할 뿐 경쟁 설명(점수의 이산성, 입력 정의 불일치, 채점기 규칙)을 배제하지 않으며, 원고의 주장에 쓰지 않는다.
- 두 실험의 대응(두 스크립트를 읽어 확인): exp-r2-empirical-noise 는 이 실험의 코호트(build_cohort)와 fold 정의(make_folds, 같은 시드)를 import 해 쓰므로 작업자 분할이 같다. 관절 집합은 같지 않다. 그 실험은 주석에 있는 17개 관절로 맞춘 입력만 쓰고(이 실험의 matched 변형과 같은 집합, 그 실험의 점검에서 gauss 가 이 실험의 matched 결과를 재현함), 이 실험의 주 분석은 예측 자세의 전체 관절을 쓴다. 따라서 아래 표는 matched 입력에서의 값이다.
- 아래 표는 3절과 같은 fold 평균 집계다(fold 를 모은 값이 아님). 열: 정규 섭동 sigma_base 0.04(현재 설정), 0.20(스윕 상한), 크기 맞춘 정규 섭동, A1(실측 잔차 재표집).

{md2(sp)}

## 5. 탐색적(결과를 본 뒤 추가)
### 5.1 탐색 격자 결과 (판정에 쓰지 않음)
탐색 격자(상한 {max(g_expl)})에서 상한에 닿은 fold 비율의 최댓값은 {t.explor_frac_folds_at_upper_bound.max():.2f}, 선택 온도의 최솟값은 {t.explor_T_min.min():g}(사전 격자 상한 {max(g_def)})이다. 사전 기준 판정식을 같은 방식으로 적용하면 주 분석 {n_expl}칸, matched {n_expl_m}칸이지만, 격자를 결과를 본 뒤 넓힌 것이므로 "보정됨"의 근거로 쓰지 않는다. 사전에 정한 격자가 아니었다.
{md2(ex)}

### 5.2 작업자 단위 구간 (주석 참조 ECE)
제조(작업자 {wk['manufacturing']}명)는 작업자 재표집 2,000회의 95% 구간이다. 값은 fold 를 모은 ECE 이며 위 3절의 fold 평균과 집계가 달라 값이 다르다(판정에 쓰지 않음).
{md2(pd.DataFrame(brow))}

요양과 물류는 작업자별 ECE 이다(구간 없음). 이미지가 30장 미만인 작업자(물류 {n_small}명, 방법마다 같은 작업자)는 표시 열에 "참고(n<30)"로 달았다. n 이 작아 작업자별 ECE 가 거의 의미 없다(사후 기준):
{md2(pwt)}

## 6. 구현자가 정한 선택(사전 기준에 없음)
5,889장 필터(상태, 분석 대상 작업자 매칭, 주석 중복 제거), matched 변형의 도입, (다) ECE 의 fold 평균 집계, 탐색 격자의 점 구성, 경계 칸 표시와 n<30 참고 표시(사후).

## 7. 재현
{run_info2(exp)}
- 이미지당 몬테카를로 채점 시간(S=128, sigma_base 0.04): 평균 {sec[0.04]:.3f}초(sigma_base 0.01 에서 0.16 까지 {sec.min():.3f}에서 {sec.max():.3f}초). 같은 실행 안 설정 간 비교용 수치이며 배포 환경의 지연으로 읽지 않는다. 출처: result/exp-r2-mc-sensitivity-v2/explor_settings.csv (해당 요약 exp-r2-mc-sensitivity-v2).
"""
    (R / exp / "summary.md").write_text(s)


def fusion():
    exp = "exp-r2-fusion-comparison-v2"
    t = pd.read_csv(R / exp / "summary_table.csv")
    m = json.loads((R / exp / "metrics.json").read_text())
    st = pd.read_csv(R / exp / "explor_strata.csv")
    wb = pd.read_csv(R / exp / "explor_boot.csv")
    cm = pd.read_csv(R / exp / "explor_cache_mismatch.csv")
    cols = ["condition", "method", "n", "pck", "pck_ci_lo", "pck_ci_hi", "OWAS_acc", "OWAS_acc_ci_lo", "OWAS_acc_ci_hi", "RULA_acc", "RULA_acc_ci_lo",
            "RULA_acc_ci_hi", "REBA_acc", "REBA_acc_ci_lo", "REBA_acc_ci_hi", "OWAS_qwk", "RULA_qwk", "REBA_qwk", "identity_subst_rate",
            "identity_subst_n", "multi_person_images", "joints_changed_vs_single"]
    ca, cb = bool(m["criterion_a"]), bool(m["criterion_b"])
    if ca and not cb:
        headline = "(a)만 충족: 단일 모델보다 나쁘지 않다. (b) 미충족: soft·max 보다 낫지 않다."
    else:
        headline = f"사전 기준 결과가 예상 문구와 다르다: criterion_a={ca}, criterion_b={cb}. 사전 문구에 맞춰 직접 서술해야 한다."
    na = t[(t.condition == "no_annotation") & t.method.isin(["single", "soft", "max", "guarded"])]
    swap_min = float(na.identity_subst_rate.min())
    n_multi = int(na.multi_person_images.iloc[0])
    # 단인·다인 층화
    base = ["single", "soft", "max", "guarded"]
    sb = st[st.method.isin(base)].copy()
    sk = st[st.method.isin(["soft_k1", "soft_k3", "soft_k10"])].copy()
    # 작업자 단위 부트스트랩
    wbn = int(wb.n_workers.max())

    def pairtab(conds, pairs):
        out = []
        for c_ in conds:
            for p_ in pairs:
                q = wb[(wb.condition == c_) & (wb.pair == p_)].set_index("stat")
                if q.empty:
                    continue
                out.append(dict(condition=c_, pair=p_, **{x: ci(q.point[x], q.ci_lo[x], q.ci_hi[x], 4) for x in ("pck", "OWAS_acc", "RULA_acc", "REBA_acc")}))
        return pd.DataFrame(out)

    main_pairs = ["guarded_minus_single", "guarded_minus_soft", "guarded_minus_max", "soft_minus_single", "max_minus_single"]
    k_pairs = ["soft_k1_minus_single", "soft_k3_minus_single", "soft_k10_minus_single", "guarded_minus_soft_k1", "guarded_minus_soft_k3", "guarded_minus_soft_k10"]
    wtab = pairtab(["same_person", "no_annotation"], main_pairs)
    ktab = pairtab(["same_person", "no_annotation"], k_pairs)
    # (b) 판정 상당: same_person 에서 guarded_minus_soft/max 의 PCK 작업자 단위 구간
    bq = wb[(wb.condition == "same_person") & (wb.stat == "pck")].set_index("pair")
    b_soft_hi = float(bq.ci_hi["guarded_minus_soft"])
    gs_single = bq.loc["guarded_minus_single"]
    ga = t[(t.condition == "same_person") & (t.method == "guarded_minus_single")].iloc[0]
    acc_ci_txt = ("모두 0 을 포함한다" if all(ga[f"{k_}_acc_ci_lo"] < 0 < ga[f"{k_}_acc_ci_hi"] for k_ in ("OWAS", "RULA", "REBA"))
                  else "일부가 0 을 포함하지 않는다(표 확인)")
    # 캐시 불일치
    cat = cm.category.value_counts()
    nums = cm[cm.category == "numeric_same_person"].diff_selected_px
    s = head2(exp, "C5", "E4 (구성요소 비교)", "고정 평가 세트 1,178장, aihub_busan_annotations",
              "1차 실행(exp-r2-fusion-comparison)은 사람 선택이 불공정해 판정에 쓰지 않음. 단인/다인 층화, 작업자 단위 구간, soft 지수 민감도는 탐색적")
    s += f"""
## 1. 목적과 결론
보호 융합(guarded)을 단일 모델, soft, max 와 정답 관절점 및 하류 행동 수준으로 비교한다(심사자 5-3). {REF} 사전 기준: targets.md "exp-r2-fusion-comparison" 과 "검수 반영 개정: exp-r2-fusion-comparison-v2".

**결론(사전 문구): {headline}** (조건 same_person 에 적용. PCK 차 구간 하한, 행동 수준 일치율 차 구간 하한은 3절.)

**융합 비교는 분석 대상자 식별이 주어졌을 때만 의미가 있다. 식별이 없으면(no_annotation) 어느 방법이든 다인 장면 {n_multi}장 중 {100 * swap_min:.1f}% 이상에서 분석 대상이 아닌 사람을 고른다.** 이 비율이 방법 간 차이보다 훨씬 크다(4절).

## 2. 설정, 실행 이력, 범위
- 1차 실행은 단일 모델과 guarded 에만 평가 이미지의 정답으로 분석 대상 사람을 골라 주고 soft 와 max 에는 주지 않았다(코드 검수 T2, T4, E2). 그래서 다시 실행했다.
- 조건 same_person: 모든 방법에서 각 모델의 검출 중 분석 대상 작업자에 해당하는 사람을 똑같이 고른 뒤 융합한다(정답으로 사람을 고르므로 배포에서 쓸 수 없는 오라클 조건이다). 조건 no_annotation: 어떤 방법에도 정답을 주지 않는다(단일 모델은 기준 모델의 주 검출, guarded 는 그 사람에 잠금, soft 와 max 는 각 모델의 주 검출). 방법마다 사람 선택 규칙이 다르다는 점이 이 조건의 한계다.
- 세 모델의 검출은 1차 실행에서 GPU 환경(bioera_E_posecmp)으로 한 번 만든 캐시(모든 사람 포함)를 다시 썼고, 융합 계산은 ace_pose/bioera_v2.py 의 기존 코드다. 평가는 bioera_mirror_test 환경.
- 불변성 점검: no_annotation 에서 정답을 바꿔도 출력이 같고 same_person 에서 선택되지 않은 사람을 흔들어도 출력이 같다(다인 장면 {m['invariance_images_checked']}장, invariance_ok={m['invariance_ok']}).
- 차이의 95% 구간은 이미지 단위 짝지은 부트스트랩 2,000회, 시드 0. 행의 method 가 a_minus_b 형태이면 차이와 그 구간이다. 판정 지표는 PCK 와 행동 수준 일치율로 사전에 고정했고 qwk 는 보고용이다. 지표와 비교쌍이 많으므로 다중성 보정은 하지 않았고 이 구간들은 탐색적 보고로 읽는다.
{run_hist(exp)}
- 범위: 고정 평가 세트 1,178장 단일 출처(aihub_busan_annotations), 작업자 {wbn}명, 시드 0. 참조는 사람이 표시한 2D 관절점이고 하류 행동 수준은 같은 채점기로 계산한 값이다. 전문가 판단도 실제 인간공학 위험도도 3D 오차도 아니다. 작업자 단위 구간은 5절(이 실험은 작업자 수가 충분해 구간을 낸다).

## 3. 사전 기준의 기계적 결과(조건 same_person 에 적용)
기준 (a) guarded 가 단일 모델보다 나쁘지 않다(PCK 차 구간 하한 > -0.01, 행동 수준 일치율 차 구간 하한 > -0.02). (b) guarded 가 soft 와 max 보다 PCK 가 높다(차이의 구간이 0 을 포함하지 않음). 마진 -0.01, -0.02 의 근거는 사전 기준에 적혀 있지 않은 연구자의 선택이다.
- (a): criterion_a={m['criterion_a']}. guarded - 단일 모델 차이의 구간 하한(이미지 단위): PCK {ga['pck_ci_lo']:.4f}(기준 > -0.01), OWAS {ga['OWAS_acc_ci_lo']:.4f}, RULA {ga['RULA_acc_ci_lo']:.4f}, REBA {ga['REBA_acc_ci_lo']:.4f}(기준 > -0.02).
- (b): criterion_b={m['criterion_b']} (soft 대비 {m['criterion_b_vs_soft']}, max 대비 {m['criterion_b_vs_max']}).
- 사전 기준 문구에 따르면 "{headline}" 에 해당한다. same_person 에서 guarded 대 soft 의 PCK 차는 음수이고 구간이 0 아래다(이미지 단위 구간은 아래 표, 작업자 단위 구간은 5절의 상한 {b_soft_hi:.4f}).
- guarded 대 단일 모델의 PCK 차는 {ga['pck']:+.4f}(이미지 단위 구간 [{ga['pck_ci_lo']:.4f}, {ga['pck_ci_hi']:.4f}], 작업자 단위 구간 [{gs_single['ci_lo']:.4f}, {gs_single['ci_hi']:.4f}])로 매우 작다. 행동 수준 일치율 차의 구간은 {acc_ci_txt}. 단일 모델보다 낫다는 주장은 하지 않고 비열등만 주장한다.

전체 표(차이 행 포함):
{md2(t[[c for c in cols if c in t.columns]], 4)}

## 4. 관찰된 이상 징후
- same_person 조건에서 soft 의 PCK 가 guarded 보다 높다(guarded_minus_soft 행의 구간이 0 아래). soft 는 거의 모든 관절을 바꾼다(joints_changed_vs_single 열).
- no_annotation 조건에서 단일 모델과 guarded 도 다인 장면의 상당 비율에서 분석 대상이 아닌 사람을 고른다(identity_subst_rate 열, 다인 장면 {n_multi}장 기준). 단일 모델과 guarded 의 값이 거의 같다. guarded 는 융합이 새로 만드는 뒤바뀜을 막는 설계이고 사람 선택 자체를 고치지는 못한다. soft 와 max 와의 차이는 두 방법이 각 모델의 주 검출을 섞어 뒤바뀜이 늘어난 데서 온다. 코호트 본분석은 데이터셋이 표시한 분석 대상 작업자를 사람 선택에 쓴다.
- soft 의 가중치는 코드에서 exp(k x 신뢰도)를 정규화한 값이며 기본 k=3 이다. 원고 방법 절에 이 상수가 적혀 있지 않다(5.2절에서 k=1, 3, 10 을 비교).

## 5. 탐색적(결과를 본 뒤 추가)
통과/미달 기준 없음. 지표 키는 explor/ 접두.

### 5.1 단인 장면과 다인 장면 층화
층은 이미지의 검출 수(기준 모델이 2명 이상 검출하면 다인)로 정의했다. 정답이 아니라 결정 시점에 아는 정보다. swap_n 은 사람 뒤바뀜 장수, swap_rate 는 그 층 안의 비율이다.
{md2(sb, 4)}

### 5.2 soft 지수 k 민감도 (k=1, 3, 10)
{md2(sk, 4)}

soft 대 단일 모델, guarded 대 soft 의 차이(작업자 단위 구간)는 아래 표다(k=3 이 기본).
{md2(ktab)}

### 5.3 작업자 단위 부트스트랩 (작업자 {wbn}명, 2,000회, 시드 0, 95% 구간)
차이 = 앞쪽 방법 - 뒤쪽 방법. 점추정 [하한, 상한].
{md2(wtab)}

### 5.4 캐시 불일치의 원인 분류
다시 검출한 단일 모델의 좌표를 코호트 자세 캐시와 이미지마다 비교했다. numeric_same_person: 재검출에서 고른 사람이 캐시의 같은 사람이고 좌표만 다름. person_selection: 고른 사람 자체가 다름. 분류별 이미지 수는 numeric_same_person {int(cat.get('numeric_same_person', 0))}장, person_selection {int(cat.get('person_selection', 0))}장이다. person_selection 의 선택된 검출은 캐시와 {cm[cm.category == 'person_selection'].diff_selected_px.max():.1f} px 어긋나고 가장 가까운 검출은 {cm[cm.category == 'person_selection'].diff_nearest_px.max():.2f} px 로 일치한다(다른 사람을 골랐다). numeric_same_person 의 좌표 차(px)는 중앙값 {nums.median():.2f}, 사분위 {nums.quantile(0.25):.2f}~{nums.quantile(0.75):.2f}, 최댓값 {nums.max():.1f} 이다. 전체 일치 이미지 비율은 {m['single_vs_batch_v1_cache_identical_share']} 이다.
원인은 GPU 대 CPU 추론의 수치 차이로 판단한다(paper/decision-log.md 2026-11-13 조사: 30장 표본에서 CPU 환경은 캐시와 30/30 일치, GPU 환경은 0/30이고 중앙값 약 2 px, 사람 선택과 행동 수준 변화는 표본에서 없음). 이 실험에서 직접 검증한 것은 아니며 6,007장 전수 확인은 아니다. 방법 간 비교는 같은 검출 위에서 짝지었으므로 이 불일치가 방법 간 우열에는 영향을 주지 않으나, 절대 수치는 이 실험 안에서만 해석한다.

## 6. 구현자가 정한 선택(사전 기준에 없음)
PCK 는 관절 수 합산 방식, 사람 뒤바뀜은 분석 대상 작업자와의 거리 기준 초과로 정의, 다인 장면은 기준 모델이 2명 이상 검출한 이미지, 비열등 마진 -0.01 과 -0.02, soft 지수 기본값 3.

## 7. 재현
{run_info2(exp)}
"""
    (R / exp / "summary.md").write_text(s)



def assumed():
    exp = "exp-r2-assumed-inputs"
    t = pd.read_csv(R / exp / "summary_table.csv")
    m = json.loads((R / exp / "metrics.json").read_text())
    inv = pd.read_csv(R / exp / "input_inventory.csv")
    p = (t.pivot_table(index=["assumption", "method"], columns="domain", values="share_action_changed", sort=False) * 100).reset_index()
    sc = (t.pivot_table(index=["assumption", "method"], columns="domain", values="share_score_changed", sort=False) * 100).reset_index()
    cls = "cls"   # 관측(observed) / 가정(assumed) / 호출자 제공 구분 열
    s = head(exp, "C2", "E5 (가정 입력의 영향)", "cohort_pose_cache 6,007장", "판정 기준 없음(기술 통계)")
    s += f"""
## 1. 목적
관측 입력과 가정 입력을 구분하고, 손목 자세와 외부 하중 가정이 점수와 행동 수준을 얼마나 바꾸는지 코호트 전체에서 잰다(심사자 5-4). 사전 기준: targets.md "exp-r2-assumed-inputs". 판정 기준은 없다.

## 2. 설정
- 코호트 6,007장 결정론 재채점, 난수 없음. 기준선(코호트 본분석 설정: 자세 전용, 손목 중립, 하중 0)이 캐시된 점수 5개 열을 {m['baseline/match_all_columns']} / {m['baseline/n']}장에서 재현했다.
- 한 번에 한 가정만 바꾼다. 하중 조건은 각 방법의 채점기 코드가 정의한 단계의 대표값을 전 코호트에 일괄 적용한 것이며, 실제로 그 하중을 든 장면이라는 뜻이 아니다.
- 이 실험은 점수 변화율만 잰다. 정답 관절점, 전문가 판단, 실제 위험과의 비교가 아니다.

## 3. 결과
행동 수준이 기준선과 달라지는 이미지 비율(%):
{md(p, 1)}

종합 점수가 달라지는 이미지 비율(%):
{md(sc, 1)}

- 행동 수준이 낮아진 경우는 없다(share_action_up 과 share_action_changed 가 같은 행: {int((abs(t.share_action_up - t.share_action_changed) < 1e-12).sum())} / {len(t)}).
- 보고 항목: 손목 가정이 행동 수준을 바꾸는 최대 비율 {100 * t[t.assumption.str.contains('wrist')].share_action_changed.max():.1f}%, 하중 가정(일괄 적용) 최대 {100 * t[t.assumption.str.startswith('load')].share_action_changed.max():.2f}%.

입력 분류(채점기 코드에서 직접 확인, 전체는 input_inventory.csv {len(inv)}행):
{md(inv.groupby(['method', cls]).size().unstack(fill_value=0).reset_index())}

## 4. 관찰된 이상 징후
- 잠정 공정 하중표는 돌봄과 식사 보조 과업에만 정의되어 있어 영역별 변화율이 크게 다르다.
- 결과는 대표 kg 값과 "최악" 손목 정의에 의존한다.

## 5. 구현자가 정한 선택(사전 기준에 없음)
하중 단계의 대표 kg 값, 손목 최악값의 정의, 반복과 충격 플래그의 조합, 입력 분류의 경계.

## 6. 재현
{run_info(exp)}
"""
    (R / exp / "summary.md").write_text(s)

def noise():
    exp = "exp-r2-empirical-noise"
    a = pd.read_csv(R / exp / "summary_A.csv")
    b = pd.read_csv(R / exp / "summary_B.csv")
    d = pd.read_csv(R / exp / "per_image.csv")
    d = d[d.target == "grand"]
    rows = []
    for (dom, me), g in d.groupby(["domain", "method"], sort=False):
        for v in ("gauss", "A1"):
            f = lambda h: ece(h[f"top_{v}"].values, (h[f"mode_{v}"] == h.target_class).values.astype(float))  # noqa: E731
            fs, ns = zip(*[(f(h), len(h)) for _, h in g.groupby("fold")])
            rows.append(dict(domain=dom, method=me, variant=v, ece_pooled=f(g), ece_fold_mean=np.mean(fs),
                             ece_fold_weighted=np.average(fs, weights=ns), ece_fold_min=min(fs), ece_fold_max=max(fs), thresh=TH[me.upper()]))
    e = pd.DataFrame(rows)
    cnt = lambda v, c: int((e[e.variant == v][c] <= e[e.variant == v].thresh).sum())  # noqa: E731
    ag = a[a.target == "grand"]
    br = lambda v: int(ag[ag.variant == v].brier_below_gauss.astype(str).eq("True").sum())  # noqa: E731
    ba = b[(b.target == "action")]
    a1 = ba[ba.distribution == "A1"]
    tau = [float(x) for s_ in a1.tau_or_q_per_fold for x in str(s_).split(";")]
    tau_sat = float(np.mean([t_ >= 0.9999 for t_ in tau]))
    # ---- 탐색적 파일 ----
    bt = pd.read_csv(R / exp / "explor_boot.csv")
    pw = pd.read_csv(R / exp / "explor_per_worker.csv")
    sw = pd.read_csv(R / exp / "explor_sigma_sweep.csv")
    need_cols(sw, FOLD_COLS, f"result/{exp}/explor_sigma_sweep.csv")
    sg = pd.read_csv(R / exp / "explor_sized_sigma.csv")
    sd = pd.read_csv(R / exp / "explor_seeds.csv")
    swg = sw[sw.target == "grand"]
    order = ["gauss0.04", "gauss0.08", "gauss0.12", "gauss0.16", "gauss0.2", "sized_gauss", "A1"]
    # 판정 집계(fold 평균)로 9칸 모두: 변형별 사전 ECE 기준 통과 칸 수와 칸별 표. pooled 는 별도 표로만 둔다.
    fe = sweep_fold("ece_fold_mean", swg, order)
    fb = sweep_fold("brier_fold_mean", swg, order)
    fe["기준"] = [TH[m] for m in fe.method]
    npass = {v: int((fe[v] <= fe["기준"]).sum()) for v in order}
    pass_tab = pd.DataFrame([dict(variant=v, ece_기준_이하_칸수=npass[v], ece_평균=fe[v].mean(), brier_평균=fb[v].mean()) for v in order])
    sweep_pooled = swg.pivot_table(index=["domain", "method"], columns="variant", values="ece", sort=False)[order].reset_index()
    sweep_pooled["기준"] = [TH[m] for m in sweep_pooled.method]
    sweep_pooled_b = swg.pivot_table(index=["domain", "method"], columns="variant", values="brier", sort=False)[order].reset_index()
    cells = swg[swg.variant.isin(["A1", "gauss0.04"])][["variant", "domain", "method", "n", "ece_fold_mean", "ece_fold_min", "ece_fold_max"]].copy()
    cells["thresh"] = [TH[m] for m in cells.method]
    cells["pass_fold_mean"] = cells.ece_fold_mean <= cells.thresh
    # per_image.csv 재계산(4절 집계 민감도 표)과 sweep 파일의 fold 평균이 같은 값인지
    chk = e.assign(variant=e.variant.replace({"gauss": "gauss0.04"}), method=e.method.str.upper()).merge(
        swg[["variant", "domain", "method", "ece_fold_mean"]], on=["variant", "domain", "method"], suffixes=("", "_sweep"))
    assert len(chk) == 18, f"per_image 대 sweep 대조 행 수 {len(chk)} (18 이어야 한다)"
    chk_diff = float((chk.ece_fold_mean - chk.ece_fold_mean_sweep).abs().max())
    assert chk_diff < 1e-9, f"per_image 재계산이 sweep 파일과 다르다({chk_diff:.2e}): ECE 구간 규칙을 확인"
    # 크기 맞춘 정규 섭동 대 A1 (풀링, 행 단위; 제조는 작업자 부트스트랩 구간)
    c = bt[(bt.analysis == "A1_vs_sized_gauss") & (bt.target == "grand") & (bt.basis == "rows")]
    crow = []
    for (dom, me), g in c.groupby(["domain", "method"], sort=False):
        q = g.set_index("stat")
        crow.append(dict(domain=dom, method=me, ece_A1=q.point["ece_A1"], ece_sized=q.point["ece_cmp"],
                         ece_diff_A1_minus_sized=ci(q.point["ece_A1_minus_cmp"], q.ci_lo["ece_A1_minus_cmp"], q.ci_hi["ece_A1_minus_cmp"]),
                         brier_A1=q.point["brier_A1"], brier_sized=q.point["brier_cmp"],
                         brier_diff_A1_minus_sized=ci(q.point["brier_A1_minus_cmp"], q.ci_lo["brier_A1_minus_cmp"], q.ci_hi["brier_A1_minus_cmp"]),
                         _elo=q.ci_lo["ece_A1_minus_cmp"], _ehi=q.ci_hi["ece_A1_minus_cmp"]))
    cm = pd.DataFrame(crow)
    n_cell = len(fe)
    # 같은 집계(fold 평균)의 부호 비교: 크기 맞춘 정규 섭동이 A1 보다 나쁘지 않은 칸
    n_le_e = int((fe["sized_gauss"] <= fe["A1"]).sum())
    n_le_b = int((fb["sized_gauss"] <= fb["A1"]).sum())
    n_both = int(((fe["sized_gauss"] <= fe["A1"]) & (fb["sized_gauss"] <= fb["A1"])).sum())
    n_brier_a1 = int((fb["A1"] < fb["gauss0.04"]).sum())
    a1_sig_worse = cm[(cm._elo > 0)]          # A1 이 유의하게 나쁜 칸(구간 있는 제조만, fold 를 모은 값 기준)
    a1_sig_better = cm[(cm._ehi < 0)]
    sig_worse_txt = ", ".join(f"{DOM_KO[r_.domain]} {r_.method}" for r_ in a1_sig_worse.itertuples()) or "없음"
    sig_better_txt = ", ".join(f"{DOM_KO[r_.domain]} {r_.method}" for r_ in a1_sig_better.itertuples()) or "없음"
    # 크기 맞춘 sigma 선택 분포
    SIG_MAX = 0.20
    sgg = sg[sg.target == "grand"]
    sdist = sgg.groupby(["domain", "method"], sort=False).sigma_base.agg(
        folds="count", median="median", at_upper_bound=lambda v: int((v >= SIG_MAX - 1e-9).sum()),  # SIG_MAX: src 의 GAUSS_SIGMAS 상한
        chosen=lambda v: ", ".join(f"{k:g} x{n}" for k, n in v.value_counts().sort_index().items())).reset_index()
    k_ub, k_all = int(sdist.at_upper_bound.sum()), int(sdist.folds.sum())
    sat_cells = ", ".join(f"{DOM_KO[r_.domain]} {r_.method}({r_.at_upper_bound}/{r_.folds})" for r_ in sdist[sdist.at_upper_bound > 0].itertuples()) or "없음"
    n_sat_cells = int((sdist.at_upper_bound > 0).sum())
    n_down = int((fe["gauss0.2"] < fe["gauss0.04"]).sum())
    # 이미지당 1회 집계(제조 3칸): 같은 이미지가 여러 fold 에서 평가되면 평균한 뒤 ECE
    io = bt[(bt.basis == "image_once") & (bt.target == "grand") & (bt.domain == "manufacturing")]
    iorow = []
    for me in dict.fromkeys(io.method):
        q1 = io[(io.method == me) & (io.analysis == "A1_vs_sized_gauss")].set_index("stat")
        q2 = io[(io.method == me) & (io.analysis == "A1_vs_gauss0.04")].set_index("stat")
        iorow.append(dict(method=me, n_images=int(q1.n_images.iloc[0]), ece_A1=ci(q1.point["ece_A1"], q1.ci_lo["ece_A1"], q1.ci_hi["ece_A1"]),
                          ece_sized_gauss=ci(q1.point["ece_cmp"], q1.ci_lo["ece_cmp"], q1.ci_hi["ece_cmp"]),
                          ece_gauss0_04=ci(q2.point["ece_cmp"], q2.ci_lo["ece_cmp"], q2.ci_hi["ece_cmp"]),
                          diff_A1_minus_sized=ci(q1.point["ece_A1_minus_cmp"], q1.ci_lo["ece_A1_minus_cmp"], q1.ci_hi["ece_A1_minus_cmp"]), thresh=TH[me]))
    iosign = {r_["method"]: np.sign(io[(io.method == r_["method"]) & (io.analysis == "A1_vs_sized_gauss") & (io.stat == "ece_A1_minus_cmp")].point.iloc[0]) for r_ in iorow}
    # 요양·물류: 이미지당 1회 집계의 구간이 없으므로 fold 평균 차이의 방향이 제조(이미지당 1회)와 같은지만 본다
    dirs = []
    for r_ in fe[fe.domain != "manufacturing"].itertuples():
        diff = r_.A1 - r_.sized_gauss
        dirs.append(dict(domain=r_.domain, method=r_.method, A1_minus_sized_fold_mean=diff, 제조와_방향_일치=bool(np.sign(diff) == iosign.get(r_.method))))
    dirs = pd.DataFrame(dirs)
    # A1 대 gauss0.04 (제조 구간)
    g4 = bt[(bt.analysis == "A1_vs_gauss0.04") & (bt.target == "grand") & (bt.basis == "rows") & (bt.domain == "manufacturing")]
    g4r = []
    for me, g in g4.groupby("method", sort=False):
        q = g.set_index("stat")
        g4r.append(dict(method=me, ece_A1=q.point["ece_A1"], ece_gauss=q.point["ece_cmp"],
                        ece_diff=ci(q.point["ece_A1_minus_cmp"], q.ci_lo["ece_A1_minus_cmp"], q.ci_hi["ece_A1_minus_cmp"]),
                        brier_diff=ci(q.point["brier_A1_minus_cmp"], q.ci_lo["brier_A1_minus_cmp"], q.ci_hi["brier_A1_minus_cmp"])))
    # 요양·물류 작업자별(종합 점수)
    pwl = pw[(pw.target == "grand") & (pw.domain != "manufacturing")].copy()
    pwtab = pwl.pivot_table(index=["domain", "method", "worker", "n_images"], columns="analysis", values="ece_A1_minus_cmp", sort=False)
    pwtab.columns = ["ece_A1_minus_" + ("gauss0.04" if "gauss0.04" in c_ else "sized") for c_ in pwtab.columns]
    pwtab = pwtab.reset_index()
    nA1b = pwtab.groupby(["domain", "method"], sort=False).agg(workers=("worker", "count"),
                                                                A1_better_than_gauss=("ece_A1_minus_gauss0.04", lambda v: int((v < 0).sum())),
                                                                A1_better_than_sized=("ece_A1_minus_sized", lambda v: int((v < 0).sum()))).reset_index()
    # 크기 맞춘 정규 섭동의 pooled 표(판정 아님)
    sz = swg[swg.variant == "sized_gauss"].copy()
    sz["thresh"] = [TH[m] for m in sz.method]
    pooled_tab = sz[["domain", "method", "n", "ece", "ece_fold_min", "ece_fold_max", "thresh"]].rename(columns={"ece": "ece_pooled"})
    pooled_tab["pass_pooled"] = pooled_tab.ece_pooled <= pooled_tab.thresh
    # 시드
    sdg = sd[sd.target == "grand"]
    sdt = sdg.groupby(["variant", "domain", "method"], sort=False).agg(ece_mean=("ece", "mean"), ece_sd=("ece", "std"), brier_sd=("brier", "std")).reset_index()
    sdp = sdt.pivot_table(index=["domain", "method"], columns="variant", values=["ece_sd", "brier_sd"], sort=False)
    sdp.columns = [f"{x}_{y}" for x, y in sdp.columns]
    sdp = sdp.reset_index()
    sd_max = float(sdt.ece_sd.max())
    # 재현 기록: 분할 파일 해시, A3 선택 온도(summary_A), A2 3분위 경계(저장 여부)
    fw_path = R / exp / "folds_workers.csv"
    fw_sha = hashlib.sha256(fw_path.read_bytes()).hexdigest()
    a3 = ag[ag.variant == "A3"][["domain", "method", "n_folds", "T_mean", "T_min", "T_max"]]
    met = json.loads((R / exp / "metrics.json").read_text())
    cut_keys = [k for k in met if re.search(r"terc|cut", k, re.I)]
    cut_txt = ("저장되지 않음(metrics.json 에 해당 키가 없고 결과 CSV 에도 해당 열이 없다)" if not cut_keys
               else "metrics.json: " + ", ".join(f"{k}={met[k]}" for k in cut_keys))
    concl = (f"크기 맞춘 정규 섭동은 9칸 중 {n_le_e}칸에서 A1 보다 ECE(fold 평균)가 나쁘지 않았다(Brier {n_le_b}칸, 둘 다 {n_both}칸). "
             f"동등성은 입증되지 않았다(제조 3칸만 차이 구간이 있고, 요양과 물류 6칸은 구간이 없다). "
             f"A1 이 유의하게(구간이 0 을 넘지 않음, fold 를 모은 값 기준) ECE 가 나쁜 칸: {sig_worse_txt}. 유의하게 좋은 칸: {sig_better_txt}. "
             f"따라서 A1 의 개선을 분포 모양(꼬리, 관절 간 상관)의 효과로 주장하지 않으며, 섭동 크기만 키워도 비슷한 개선이 난다는 것까지 말한다")
    s = head2(exp, "C5", "E2 (보정 타당성, 2차 사전 기준 A 와 B)", "cohort_pose_cache, aihub_busan_annotations 5,889장",
              "설계를 exp-r2-indep-calibration 결과를 같은 이미지에서 본 뒤에 만들었다(저자 승인 예외, paper/audit/exceptions.json). 7절 이후는 결과를 본 뒤 추가한 탐색적 분석")
    s += f"""
## 1. 목적과 결론
exp-r2-indep-calibration 에서 현재 정규 섭동이 주석 참조 점수에 대해 과대확신임이 드러났다. 섭동을 실측 잔차의 통째 재표집으로 바꾸고(A), 적합 예측 집합을 더한다(B). {REF_ANN} 사전 기준: targets.md "2차 사전 기준". 이 요약의 비교는 자체 방법족(정규 섭동, 실측 잔차 재표집, 온도 조정) 안에서의 비교다.

이 요약의 기준 A 칸 수는 모두 같은 집계(fold 평균)다. fold 를 모은(pooled) 값과 가중 평균은 4절 끝의 별도 표에만 두고 판정에 쓰지 않는다.

- 기준 A(판정 집계는 fold 평균): A1 은 9칸 중 {npass['A1']}칸이 ECE 기준 이하로 7칸 기준에 {'충족' if npass['A1'] >= 7 else '미달'}이다. 정규 섭동(gauss0.04)은 {npass['gauss0.04']}칸이다. Brier(fold 평균)가 gauss0.04 보다 낮은 A1 칸은 {n_brier_a1} / 9다.
- 기준 B: 모은 포함률 0.88 이상 {int((a1.coverage >= 0.88).sum())} / 9, 고정 폭보다 작은 집합 {int(a1.smaller_than_fixed.astype(str).eq('True').sum())} / 9. 임계값 tau 가 {100 * tau_sat:.0f}% 의 fold 에서 1.0 에 포화해 집합이 넓다. 정보가 적은 결과이며 기준 B 를 보정의 근거로 쓰지 않는다.
- 탐색적 핵심 결과(7, 8절, 결과를 본 뒤 추가): {concl}.
- 한계: sigma 격자 상한(0.20)을 고른 fold 가 {k_ub} / {k_all}({100 * k_ub / k_all:.0f}%)이고, 상한을 고른 fold 가 있는 칸이 {n_sat_cells} / 9칸이다. 상한 포화: 더 큰 sigma 는 시험하지 않았다. 따라서 이 칸들의 크기 맞춘 정규 섭동 결과는 격자 안 최선이며 최적 sigma 가 아닐 수 있다.

## 2. 반드시 알아야 할 사실과 실행 이력
- 이 설계는 1차 실험의 결과를 같은 이미지에서 본 뒤에 만들었다. 한 번도 보지 않은 자료는 남아 있지 않다. 완화책으로 잔차 기증, 임계값, 온도 적합은 학습 작업자에서만 하고 평가는 다른 작업자에서만 했다(30개 fold 모두 기증 작업자와 평가 작업자가 겹치지 않음을 실행 중 단언, folds_workers.csv). 이 완화책이 같은 자료를 이미 본 효과를 없애지는 못한다. 따라서 이 실험 전체를 탐색적(결과를 본 뒤 설계) 근거로 취급하고 확증 주장에 쓰지 않는다.
- 관절 집합은 주석에 맞췄다(예측 자세에서 귀, 새끼발가락, 발뒤꿈치 제외).
- 검증 범위는 관절점 위치 오차뿐이다. 시점, 평면 밖 자세, 영상에 없는 입력, 평가자 차이는 포함하지 않는다.
{run_hist(exp)}
{scope(ref=False)}

## 3. 설정
- gauss: 현재 정규 섭동(sigma 0.04, S 128). A1: 같은 영역 학습 작업자의 (주석 - 예측) 잔차 벡터를 통째로 128개 복원 추출해 더함, 온도 조정 없음. A2: 기증 이미지를 평균 신뢰도 3분위가 같은 것으로 제한. A3: A1 에 학습 작업자에서 맞춘 온도(0.1~50).
- 점검: 잔차를 0 으로 두면 최빈값이 결정론 점수와 100% 같다. gauss 는 exp-r2-indep-calibration 의 matched 열을 재현한다.
- 제조 영역의 n 은 20회 반복 분할의 평가 이미지를 모은 수라 같은 이미지가 여러 번 들어간다. 고유 이미지와 작업자 수는 explor_boot.csv 의 n_images, n_workers 열. 제조 3칸의 이미지당 1회 집계는 8절 끝 표.
- ECE 는 10구간 등폭이다.

## 4. 기준 A 의 기계적 결과: 판정은 fold 평균(9칸 모두)
기준: A1 의 작업자 분리 ECE 가 기준 이하인 칸이 9칸 중 7칸 이상이고 Brier 가 gauss 보다 낮은 칸이 7칸 이상.
사전 기준 문구는 ECE 를 어느 집계로 판정하는지 적지 않았다. 판정은 1차 실험(exp-r2-indep-calibration)과 같은 fold 평균으로 한다(paper/decision-log.md). 이 절과 8절의 표는 explor_sigma_sweep.csv 의 `ece_fold_mean`, `brier_fold_mean` 열(제조 포함 9칸 모두)에서 읽었다.

변형별 사전 ECE 기준(OWAS 0.05, RULA 0.07, REBA 0.10) 통과 칸 수(fold 평균, 9칸 모두):
{md2(pass_tab)}

칸별 ECE(fold 평균, 종합 점수 대상, A1 과 gauss0.04):
{md2(cells)}

### 집계 민감도(fold 를 모은 값과 가중 평균, 판정에 쓰지 않음)
per_image.csv 에서 다시 계산했다. fold 평균은 위 표와 같은 집계이며 per_image.csv 재계산과 sweep 파일 값의 최대 절대 차이는 {chk_diff:.2e} 다.

| 집계 | A1 기준 이하 칸 | gauss 기준 이하 칸 |
|---|---|---|
| fold 평균(판정) | {cnt('A1', 'ece_fold_mean')} / 9 | {cnt('gauss', 'ece_fold_mean')} / 9 |
| fold 크기 가중 평균 | {cnt('A1', 'ece_fold_weighted')} / 9 | {cnt('gauss', 'ece_fold_weighted')} / 9 |
| fold 를 모음(pooled) | {cnt('A1', 'ece_pooled')} / 9 | {cnt('gauss', 'ece_pooled')} / 9 |

- Brier 가 gauss 보다 낮은 칸(fold 를 모은 값, summary_A.csv): A1 {br('A1')} / 9, A2 {br('A2')} / 9, A3 {br('A3')} / 9. 이 부호 계산의 크기와 구간은 7절.
- 스크립트가 낸 pooled 기준 통과 칸 수: A1 {int(ag[ag.variant == 'A1']['pass'].astype(str).eq('True').sum())}, A2 {int(ag[ag.variant == 'A2']['pass'].astype(str).eq('True').sum())}, A3 {int(ag[ag.variant == 'A3']['pass'].astype(str).eq('True').sum())} / 9.

칸별 pooled 와 가중 ECE(별도 표, 판정에 쓰지 않음):
{md2(e)}

A1 과 gauss 의 정확도, 확신도, Brier(fold 를 모은 값, 별도 표):
{md2(ag[ag.variant.isin(['gauss', 'A1'])][['variant', 'domain', 'method', 'n', 'n_folds', 'mode_acc', 'mean_top_conf', 'ece', 'brier', 'coverage_5_95']])}

## 5. 기준 B 의 기계적 결과(행동 수준 대상, 목표 포함률 0.90)
기준: 모은 포함률이 9칸 모두 0.88 이상이고, 평균 집합 크기가 고정 폭 집합보다 작은 칸이 7칸 이상.
- 포함률 0.88 이상: {int((a1.coverage >= 0.88).sum())} / 9. 고정 폭보다 작은 칸: {int(a1.smaller_than_fixed.astype(str).eq('True').sum())} / 9.
- 사전 기준 문구에 따르면 "유효하나 고정 폭보다 낫지 않음"에 해당한다.
- tau 포화: A1 행동 수준 9칸의 fold 별 tau {len(tau)}개 중 {100 * tau_sat:.0f}% 가 1.0 이다. tau 가 1.0 이면 누적 확률이 1 이 될 때까지 집합을 넓히므로 포함률이 명목 0.90 을 크게 넘고(포함률 최솟값 {a1.coverage.min():.3f}) 집합이 넓어진다. 따라서 B 는 "유효하지만 정보가 적은" 보수적 집합이며 보정 타당성의 증거로 쓰지 않는다. tau 가 1.0 이 되는 원인(비순응 점수 정의, 질량 0 수준에 더한 작은 값, 동률 처리)은 분리하지 않았다(한계, 10절).

{md2(ba[['distribution', 'domain', 'method', 'n', 'coverage', 'coverage_fold_min', 'coverage_fold_max', 'mean_set_size', 'singleton_share', 'singleton_accuracy', 'multi_action_level_share']])}

## 6. 관찰된 이상 징후
- A1 은 최빈값 정확도를 거의 바꾸지 않고 최상위 확신도를 낮춘다.
- 작업자가 적은 영역(요양, 물류)은 fold 범위가 넓다(4절 ece_fold_min, ece_fold_max).
- A3(온도 조정)은 A1 보다 통과 칸이 적다. 원인은 분석하지 않았다.

## 7. 탐색적(결과를 본 뒤 추가): 작업자 단위 부트스트랩과 A1 대 gauss
통과/미달 기준 없음. 제조(작업자 {n_workers()['manufacturing']}명)만 구간을 낸다(작업자 재표집 2,000회, 시드 0, 백분위 95%). 값은 fold 를 모은 값이다. 차이는 A1 - gauss 이므로 음수면 A1 이 낫다.
{md2(pd.DataFrame(g4r))}

요양과 물류는 작업자별 값이다. 아래는 작업자 한 명씩 제외한 fold(= 그 작업자가 평가 대상)에서 A1 이 gauss 와 크기 맞춘 정규 섭동보다 ECE 가 낮은 작업자 수다.
{md2(nA1b)}

작업자별 ECE 차이(A1 - 비교 대상, 음수면 A1 이 낫다):
{md2(pwtab)}

## 8. 탐색적(결과를 본 뒤 추가): sigma 스윕과 크기 맞춘 정규 섭동
설계: 정규 섭동의 sigma_base 를 0.04, 0.08, 0.12, 0.16, 0.20 으로 바꿔 같은 주석 참조 기준으로 채점한다. "크기 맞춘 정규 섭동"은 fold 마다 학습 작업자의 주석 관절점 기준 Brier 가 가장 낮은 sigma_base 를 골라 평가 작업자에 적용한 것이다(sigma 격자의 상한은 0.20). 해석 규칙은 실행 전에 고정했다(targets.md "심사 반영 추가 분석"). 그 규칙의 "과반"(5칸 이상)은 구현자의 구체화였고, 이 요약은 그 규칙으로 원인을 귀속하지 않고 칸 수만 보고한다.

sigma 별 ECE(종합 점수, fold 평균, 9칸 모두, 기준 열은 사전 ECE 기준):
{md2(fe)}

sigma 별 Brier(종합 점수, fold 평균):
{md2(fb)}

- sigma_base 를 0.04 에서 0.20 으로 올렸을 때 ECE(fold 평균)가 줄어든 칸: {n_down} / 9. 기준 이하 칸은 {npass['gauss0.04']} 칸에서 {npass['gauss0.2']} 칸이 된다.
- fold 가 고른 sigma 의 분포(종합 점수). 상한(0.20)을 고른 fold 수를 함께 적는다. 전체 {k_ub} / {k_all} fold({100 * k_ub / k_all:.0f}%), 상한을 고른 칸: {sat_cells}. 상한 포화: 더 큰 sigma 는 시험하지 않았다(최적점이 격자 밖일 수 있다).
{md2(sdist)}

- 크기 맞춘 정규 섭동의 ECE 가 A1 이하인 칸(fold 평균) {n_le_e} / {n_cell}, Brier 가 A1 이하인 칸 {n_le_b} / {n_cell}, 둘 다인 칸 {n_both} / {n_cell}.
- A1 이 유의하게(구간이 0 을 넘지 않음) ECE 가 나쁜 칸: {sig_worse_txt}. A1 이 유의하게 ECE 가 좋은 칸: {sig_better_txt}(구간은 제조만 있다, 아래 pooled 표의 구간).
- 결론: {concl}.
- 쓸 수 있는 문장의 범위: 정규 섭동의 크기를 학습 작업자의 주석 관절점 기준으로 고르면 주석 참조 기준 오차가 크게 준다. 이 크기 조정만으로 A1 과 비슷한 수준이 난다. 꼬리나 관절 간 상관 같은 분포 모양의 효과, 통째 재표집이 원인이라는 주장은 하지 않는다.

### 제조 3칸의 이미지당 1회 집계 ECE(explor_boot.csv 의 basis=image_once)
같은 이미지가 여러 fold 에서 평가되면 fold 들의 값을 평균한 뒤 ECE 를 계산하고(n_images 열), 구간은 작업자 재표집 2,000회의 95%다. 차이는 A1 - 크기 맞춘 정규 섭동(양수면 크기 맞춘 쪽이 낫다).
{md2(pd.DataFrame(iorow))}

요양과 물류는 이미지당 1회 집계의 구간이 없어 방향 일치로만 서술한다. 아래는 fold 평균 차이(A1 - 크기 맞춘 정규 섭동)의 부호가 제조(이미지당 1회) 같은 방법의 부호와 같은지다.
{md2(dirs)}

### pooled 값(fold 를 모은 값, 별도 표, 판정에 쓰지 않음)
크기 맞춘 정규 섭동 대 A1(fold 를 모은 값, 차이 = A1 - 크기 맞춘 정규 섭동이므로 양수면 크기 맞춘 정규 섭동이 낫다, 구간은 제조만):
{md2(cm.drop(columns=[c_ for c_ in cm.columns if c_.startswith('_')]))}

sigma 별 pooled ECE:
{md2(sweep_pooled)}

sigma 별 pooled Brier:
{md2(sweep_pooled_b)}

크기 맞춘 정규 섭동의 pooled ECE 와 기준 통과(탐색적, 기준 통과로 주장하지 않음):
{md2(pooled_tab)}

## 9. 탐색적(결과를 본 뒤 추가): 몬테카를로 시드 0, 1, 2 의 표준편차
A1 의 잔차 표본 추출과 gauss 의 섭동 난수를 시드 0, 1, 2 로 바꿔 ECE 와 Brier 를 다시 계산했다(종합 점수). 표준편차는 시드 3개의 표본 표준편차다. ECE 표준편차의 최댓값은 {sd_max:.4f} 이다. 판정 기준과의 거리 ({sorted(TH.values())})에 견주어 읽는다.
{md2(sdp, 4)}

## 10. 한계
- 구성요소 분리(ablation)는 하지 않았다. A1 의 구성요소(통째 재표집, 영역별 기증, 관절 간 상관)와 A2, A3 가 A1 보다 나쁜 이유는 분석하지 않았다.
- 외부 불확실성 기준선(정확도 기반 온도 조정, 점수에 직접 적용한 split conformal)은 없다. 비교는 자체 방법족 안에서의 비교로 한정한다.
- 상한 포화: sigma 격자 상한(0.20)에 닿은 fold 가 {k_ub} / {k_all}이며 더 큰 sigma 는 시험하지 않았다.
- 동등성은 입증되지 않았다(제조의 차이 구간만, 요양과 물류는 구간 없음, 동등성 마진은 사전에 정하지 않았다).
- 기준 B: tau 포화의 원인(비순응 점수 정의, 질량 0 수준에 더한 가산값, 동률 처리)을 분리하지 않았고 명목 수준을 바꾼 포함률 곡선도 없다. 기준 B 를 보정의 근거로 쓰지 않는다.
- sigma 선택에는 해당 영역의 주석 관절점이 필요하다(학습 작업자의 주석). 필요한 주석 규모는 추정하지 않았다.

## 11. 구현자가 정한 선택(사전 기준에 없음)
기준 A 의 fold 평균 판정과 pooled 병기, 예측 집합의 동률 처리와 질량 0 수준에 더한 작은 값, 고정 폭 집합의 q 정의, 몸통 길이를 쓸 수 없는 7장의 기증 제외, 보정용 이미지의 자기 작업자 기증 제외, A2 의 3분위 경계, 온도 격자, 크기 맞춘 정규 섭동의 sigma 격자와 "과반"의 구체화.

## 12. 재현
{run_info2(exp)}
- 분할 파일 folds_workers.csv 의 sha256: `{fw_sha}`
- A2 3분위 경계: {cut_txt}.
- A3 선택 온도(summary_A.csv, 종합 점수, fold 별 값의 평균과 범위):
{md2(a3)}
"""
    (R / exp / "summary.md").write_text(s)


def selective():
    exp = "exp-r2-selective"
    t = pd.read_csv(R / exp / "summary.csv")
    cells = pd.read_csv(R / exp / "explor_cells.csv")
    curves = pd.read_csv(R / exp / "explor_fold_curves.csv")
    mn = pd.read_csv(R / exp / "explor_minacc.csv")
    mnc = pd.read_csv(R / exp / "explor_minacc_cells.csv")
    bt = pd.read_csv(R / exp / "explor_boot.csv")
    pw = pd.read_csv(R / exp / "explor_per_worker.csv")
    k = lambda src, tg: t[(t.source == src) & (t.target == tg)]  # noqa: E731
    ok = lambda q: q[q.acc_acc_fold_mean >= 0.93]  # noqa: E731
    a = k("A1", "action")
    n_excl = len(ok(a))                                                  # 수용 없음 fold 를 평균에서 뺀 규칙
    n_cons = len(ok(a)[ok(a).n_folds_no_acceptance == 0])                # 수용 없음 fold 가 있는 칸을 뺀 규칙
    g_excl = len(ok(k("A1", "grand")))
    g_cons = len(ok(k("A1", "grand"))[ok(k("A1", "grand")).n_folds_no_acceptance == 0])

    def main_set(src, tg):
        """주장에 쓸 칸: 모든 fold 에서 수용이 있고 fold 평균 수용 비율 0.20 이상."""
        q = k(src, tg).copy()
        q["mask"] = (q.n_folds_no_acceptance == 0) & (q.share_fold_mean >= 0.20)
        return q

    def main_table(src, tg):
        q = main_set(src, tg)
        c = cells[(cells.src == src) & (cells.target == tg)].set_index(["domain", "method"])
        bm = bt[(bt.analysis == f"selective_{src}") & (bt.target == tg) & (bt.basis == "rows") & (bt.domain == "manufacturing")]
        out = []
        for r_ in q[q["mask"]].itertuples():
            cc = c.loc[(r_.domain, r_.method)]
            ac = ap = ""
            if r_.domain == "manufacturing":
                x = bm[(bm.method == r_.method)].set_index("stat")
                ac = ci(x.point["accepted_acc"], x.ci_lo["accepted_acc"], x.ci_hi["accepted_acc"])
                ap = ci(x.point["accepted_share"], x.ci_lo["accepted_share"], x.ci_hi["accepted_share"])
            out.append(dict(domain=r_.domain, method=r_.method, accept_folds=f"{int(cc.folds_with_acceptance)}/{int(r_.n_folds)}",
                            accepted_acc_fold_mean=r_.acc_acc_fold_mean, worker_boot_95=ac, worst_fold=cc.acc_acc_worst_fold,
                            accepted_share_fold_mean=r_.share_fold_mean, share_worker_boot_95=ap,
                            unaccepted_share_human_review=cc.unaccepted_share, rejected_acc=r_.rej_acc_fold_mean,
                            overall_acc=r_.overall_acc, majority_baseline=cc.majority_share_heldout))
        return pd.DataFrame(out), q

    mt_a, qa = main_table("A1", "action")
    mt_g, qg = main_table("A1", "grand")
    mt_gauss, qgs = main_table("gauss", "action")
    ex = qa[~qa["mask"]][["domain", "method", "n_folds", "n_folds_no_acceptance", "acc_acc_fold_mean", "share_fold_mean"]].copy()
    def why(r_):
        w = []
        if r_.n_folds_no_acceptance > 0:
            w.append(f"수용 없는 fold {int(r_.n_folds_no_acceptance)}개")
        if r_.share_fold_mean < 0.20:
            w.append("수용 비율 0.20 미만")
        return "; ".join(w)
    ex["excluded_because"] = [why(r_) for r_ in ex.itertuples()]
    # 요양·물류 작업자별
    pwa = pw[(pw.analysis == "selective_A1") & (pw.target == "action") & (pw.domain != "manufacturing")]
    # 고정 수용 비율과 AURC (fold 평균)
    cu = curves[curves.target == "action"].groupby(["src", "domain", "method"], sort=False).mean(numeric_only=True).reset_index()
    ac_cols = [c for c in cu.columns if c.startswith("acc_at_")]
    fixed = []
    for (dom, me), _ in cu.groupby(["domain", "method"], sort=False):
        x = cu[(cu.domain == dom) & (cu.method == me)].set_index("src")
        row = dict(domain=dom, method=me, aurc_A1=x.aurc["A1"], aurc_gauss=x.aurc["gauss"], aurc_random=x.aurc_random["A1"])
        for c in ac_cols:
            row[c.replace("acc_at_", "A1_acc@")] = x[c]["A1"]
            row[c.replace("acc_at_", "gauss_acc@")] = x[c]["gauss"]
        row["random_acc"] = x.acc_random["A1"]
        fixed.append(row)
    fx = pd.DataFrame(fixed)
    n_aurc_gauss = int((fx.aurc_A1 < fx.aurc_gauss).sum())
    n_aurc_rand = int((fx.aurc_A1 < fx.aurc_random).sum())
    n_aurc_gr = int((fx.aurc_gauss < fx.aurc_random).sum())
    acc_cols_a1 = [c for c in fx.columns if c.startswith("A1_acc@")]
    n_a1_beats_rand = int((fx[acc_cols_a1].gt(fx.random_acc, axis=0)).to_numpy().sum())
    fx_show = fx[["domain", "method", "aurc_A1", "aurc_gauss", "aurc_random", "random_acc"] + acc_cols_a1 + [c for c in fx.columns if c.startswith("gauss_acc@")]]
    # 학습 목표 정확도 민감도
    m2 = mnc.copy()
    m2["main_set"] = (m2.n_folds_no_acceptance == 0) & (m2.share_fold_mean >= 0.20) & (m2.acc_acc_fold_mean >= 0.93)
    ms = (m2.groupby(["min_acc", "source", "target"], sort=False)
          .agg(cells_acc_ge_0_93_excl_rule=("pass", "sum"), cells_main_set_acc_ge_0_93=("main_set", "sum"),
               share_fold_mean_all_cells=("share_fold_mean", "mean")).reset_index())
    ms[["cells_acc_ge_0_93_excl_rule", "cells_main_set_acc_ge_0_93"]] = ms[["cells_acc_ge_0_93_excl_rule", "cells_main_set_acc_ge_0_93"]].astype(int)
    ms_a1 = ms[(ms.source == "A1") & (ms.target == "action")]
    # 오차 구조
    es = cells[(cells.src == "A1") & (cells.target == "action")][["domain", "method", "n_levels_present", "majority_share_heldout", "mode_acc",
                                                                   "adjacent_error_share", "folds_with_acceptance", "n_folds"]]
    reba = qa[qa.method == "REBA"]
    reba_a = cells[(cells.src == "A1") & (cells.target == "action") & (cells.method == "REBA")]
    s = head2(exp, "C5", "E2 (3차 사전 기준, 선택적 정확도)", "cohort_pose_cache, aihub_busan_annotations 5,889장",
              "exp-r2-empirical-noise 에서 관련 수치를 본 뒤 사전 기준으로 옮긴 검증. 탐색적 확인이며 충족을 주장하지 않음")
    s += f"""
## 1. 목적과 결론
확신도가 높은 판정만 수용했을 때의 정확도가 처음 보는 작업자에서도 유지되는지 확인한다(심사자 6). {REF} 사전 기준: targets.md "3차 사전 기준".

**사전 기준(7칸)은 집계 규칙에 따라 {n_excl}칸과 {n_cons}칸으로 갈려 충족을 주장하지 않는다. 이 실험은 탐색적 확인이다.** (행동 수준 A1: 수용 없음 fold 를 평균에서 빼면 {n_excl}칸, 수용 없음 fold 가 있는 칸을 빼면 {n_cons}칸. 종합 점수 A1: {g_excl}칸과 {g_cons}칸.)

원고에 쓸 수 있는 주장은 아래 3절의 칸(모든 fold 에서 수용이 있고 fold 평균 수용 비율이 0.20 이상인 칸)에 한정한다. 행동 수준 A1 에서 그런 칸은 {int(qa['mask'].sum())}칸이고, gauss 확신도로는 {int(qgs['mask'].sum())}칸이다. REBA 는 선택적 수용이 적용되지 않는다(6절).

## 2. 반드시 알아야 할 사실, 실행 이력, 범위
- exp-r2-empirical-noise 에서 원소 1개 집합의 정확도와 위험-포함 곡선을 이미 본 뒤에 이 기준을 적었다(targets.md 에 명시). 실측 오차 모형 자체가 1차 결과를 본 뒤의 설계라는 점도 그대로 적용된다. 학습 목표 0.96 과 판정선 0.93 도 그 관찰에서 나왔을 수 있다(5절에서 목표를 바꿔 민감도를 본다).
- 판정 집계는 fold 평균(사전 기준에 명시)이고 모은 값과 fold 범위를 함께 본다.
{run_hist(exp)}
{scope()}

## 3. 주 표: 모든 fold 에서 수용이 있고 fold 평균 수용 비율 0.20 이상인 칸
수용 정확도는 학습 작업자에서 고른 임계값을 처음 보는 작업자에 그대로 적용한 값이다. "수용되지 않은 비율"은 자동 판정을 쓰지 못해 사람이 검토해야 하는 몫(사람 검토 부담)이다. 다수결 기준선은 평가 작업자에서 가장 흔한 행동 수준만 항상 답했을 때의 정확도다. 구간은 제조의 작업자 단위 부트스트랩이고 요양과 물류는 아래 작업자별 표를 본다.

행동 수준 대상, A1 확신도(주):
{md2(mt_a)}

- 다수결 기준선과의 차이(수용 정확도 - 기준선, 주 표의 칸): {', '.join(f'{r_.domain}/{r_.method} {r_.accepted_acc_fold_mean - r_.majority_baseline:+.3f}' for r_ in mt_a.itertuples())}. 기준선이 이미 높은 RULA 칸은 확신도 선택으로 얻는 이득이 작다. 오답은 대부분 인접 수준이다(5.3절 adjacent_error_share 최솟값 {cells[(cells.src == 'A1') & (cells.target == 'action')].adjacent_error_share.min():.3f}).

행동 수준 대상, 주 표에서 빠진 칸(A1):
{md2(ex)}

요양과 물류의 작업자별 값(행동 수준, A1, 모든 칸):
{md2(pwa[['domain', 'method', 'worker', 'n_images', 'accepted_acc', 'accepted_share']])}

종합 점수 대상(부), A1 확신도, 같은 조건의 칸:
{md2(mt_g)}

비교: 행동 수준 대상, 현재 정규 섭동(gauss) 확신도로 같은 조건을 만족하는 칸 {int(qgs['mask'].sum())}칸:
{md2(mt_gauss)}

## 4. 사전 기준의 기계적 결과(두 규칙)
기준: 수용 정확도의 fold 평균이 0.93 이상인 칸이 9칸 중 7칸 이상. 그 칸들의 수용 비율 fold 평균이 0.20 미만이면 "성립하나 적용 범위가 좁다".
- 행동 수준(주), A1: 수용 없음 fold 를 평균에서 제외하면 {n_excl} / 9칸, 수용 없음 fold 가 있는 칸을 통과에서 빼면 {n_cons} / 9칸. gauss: {len(ok(k('gauss', 'action')))} / 9칸.
- 종합 점수(부), A1: {g_excl} / 9칸과 {g_cons} / 9칸. gauss: {len(ok(k('gauss', 'grand')))} / 9칸.
- 사전 기준이 "수용 없음" fold 의 집계를 정하지 않았다(검수 E3). 결과를 본 뒤 한쪽을 고르지 않고 두 결과를 함께 보고하며 충족을 주장하지 않는다(targets.md "검수 반영 개정 2").
- 수용 비율 문구는 모호하다. 스크립트는 통과 칸의 수용 비율 평균({ok(a).share_fold_mean.mean():.3f})으로 판정했으나 칸별로 보면 통과 칸 중 수용 비율이 0.20 미만인 칸이 {int((ok(a).share_fold_mean < 0.20).sum())}칸이다. 그래서 주 표는 칸별 조건(3절)을 쓴다.

행동 수준 대상 전체 칸(A1, 수용 없음 fold 수와 최악 fold 포함):
{md2(k('A1', 'action')[['domain', 'method', 'n_folds', 'n_folds_no_acceptance', 'acc_acc_fold_mean', 'acc_acc_fold_min', 'acc_acc_fold_max', 'acc_acc_pooled', 'share_fold_mean', 'share_pooled', 'rej_acc_fold_mean', 'overall_acc', 't_mean', 't_min', 't_max', 'pass']])}

행동 수준 대상 전체 칸(gauss). 수용 정확도가 빈 칸은 모든 fold 에서 조건을 만족하는 임계값이 없어 수용이 일어나지 않은 칸이다:
{md2(k('gauss', 'action')[['domain', 'method', 'n_folds', 'n_folds_no_acceptance', 'acc_acc_fold_mean', 'acc_acc_fold_min', 'acc_acc_fold_max', 'share_fold_mean', 'overall_acc', 'pass']])}

## 5. 탐색적(결과를 본 뒤 추가): 고정 수용 비율, 학습 목표 민감도, 오차 구조
이하 분석은 통과/미달 기준이 없다. 지표 키는 explor/ 접두.

### 5.1 고정 수용 비율에서의 정확도와 AURC (행동 수준, fold 평균)
확신도 순위의 품질을 임계값 선택과 분리해 본다. 확신도가 높은 순으로 상위 10%, 20%, 30%, 50%, 80% 를 수용했을 때의 정확도(A1@x, gauss@x)와 위험-포함 곡선 면적 AURC(작을수록 좋음), 무작위로 같은 비율을 수용했을 때(전체 정확도와 같음)를 비교한다.
- AURC 가 A1 < gauss 인 칸: {n_aurc_gauss} / 9. A1 < 무작위인 칸: {n_aurc_rand} / 9. gauss < 무작위인 칸: {n_aurc_gr} / 9.
- 고정 비율 5개 x 9칸 = 45개 값 중 A1 의 수용 정확도가 무작위(전체 정확도)보다 높은 값: {n_a1_beats_rand} 개.
{md2(fx_show)}

### 5.2 학습 목표 정확도 민감도 (목표 0.93, 0.95, 0.96, 0.97)
임계값 선택의 학습 목표만 바꾸고 판정선(수용 정확도 fold 평균 0.93)은 그대로 둔 결과다. cells_main_set_acc_ge_0_93 는 모든 fold 수용, 수용 비율 0.20 이상, 수용 정확도 0.93 이상을 함께 만족하는 칸 수다.
{md2(ms)}

명시적 판정(explor_minacc.csv, 수용 없음 fold 를 뺀 규칙 대 빼지 않은 규칙의 갈림, 행동 수준 A1):
{md2(mn[(mn.source == 'A1') & (mn.target == 'action')][['min_acc', 'cells_pass', 'cells_pass_all_folds_accept', 'passing_cells_share_lt_0.20', 'verdict', 'share_fold_mean_all_cells']])}

### 5.3 오차 구조: 인접 수준 오답과 다수결 기준선 (행동 수준, A1)
- 인접 수준 오답 비율은 틀린 판정 중 정답과 행동 수준이 1단계 차이인 비율이다. 값이 크면 오답이 먼 수준이 아니라 이웃 수준으로 번진다.
- 다수결 기준선은 가장 흔한 행동 수준만 답하는 정확도다. 기준선이 높은 칸(예: 수준 분포가 한쪽으로 치우친 칸)에서는 높은 정확도가 확신도의 신호가 아니라 기저율에서 올 수 있다. majority_share_heldout 과 mode_acc, 3절의 수용 정확도를 견주어 읽는다.
{md2(es)}

## 6. REBA 는 선택적 수용이 적용되지 않음
REBA 3칸(A1, 행동 수준)에서 모든 fold 수용 {int((reba.n_folds_no_acceptance == 0).sum())}칸, 수용 비율 0.20 이상 {int((reba.share_fold_mean >= 0.20).sum())}칸이다. 제조 REBA 는 {int(reba_a[reba_a.domain == 'manufacturing'].folds_with_acceptance.iloc[0])} / {int(reba_a[reba_a.domain == 'manufacturing'].n_folds.iloc[0])} fold 에서만 수용이 있어 표의 수용 정확도가 소수 fold 값이다. 요양과 물류 REBA 는 수용 정확도가 높게 보이나 수용 비율이 0.20 미만이라 대부분을 사람이 검토해야 한다. 따라서 REBA 에는 선택적 수용을 적용하지 않는 것으로 서술한다(수준 수가 4개이고 다수결 기준선도 낮다: 5.3).

## 7. 관찰된 이상 징후와 한계
- 수용 없는 fold 가 많은 칸이 있다(n_folds_no_acceptance 열). 그런 칸의 수용 정확도 평균은 소수 fold 값이다.
- gauss 확신도로는 다수 칸에서 기준을 만족하는 t 가 없다.
- 방법에 따라 수용 비율 차이가 크다. 수준 수, 경계 밀도, 확신도 분포와의 관계는 가설이며 검정하지 않았다.
- 정확도는 "정답 관절점 기준 채점 일치도"이며 현장 위험 분류 정확도가 아니다.

## 8. 구현자가 정한 선택(사전 기준에 없음)
수용 없는 fold 의 처리, 수용 비율 문구의 판정 방식, 임계값 후보(학습 확신도의 고유값), 최빈값 동률 시 낮은 수준 선택, 주 표의 칸 조건(모든 fold 수용, 수용 비율 0.20 이상), 5.2 의 main_set 정의.

## 9. 재현
{run_info2(exp)}
"""
    (R / exp / "summary.md").write_text(s)



# ---- 3차(2026-11-14): 섭동 모형 검증과 전문가 일치. 위 여섯 함수와 공통 도구는 건드리지 않는다. ----
PV, EA_ = "exp-r2-perturbation-validation", "exp-r2-expert-agreement"
LEVELS_EA = {"OWAS": [1, 2, 3, 4], "RULA": [1, 2, 3, 4], "REBA": [0, 1, 2, 3, 4]}   # src/exp_r2_expert_agreement.py 의 LEVELS 와 같다
PD = ("all",) + DOMS
PD_KO = {"all": "전체", **DOM_KO}
SIGS = ("0.01", "0.02", "0.04", "0.08", "0.16")
SCNT = (32, 64, 128, 256, 512)
ANGS = ("trunk", "neck", "upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r", "knee_l", "knee_r")
POUT = ("owas_action", "rula_action", "reba_action", "rula_grand", "reba_grand")
PMETH = ("owas", "rula", "reba")
NAN = float("nan")


def _pk(M, k, nd=3):
    """metrics 키 k 의 '점추정 [하한, 상한]'(구간 키가 없으면 점추정만). 키가 없으면 KeyError: 표의 모든 수치는 metrics.json 에 있다."""
    return ci(M[k + "/point"], M.get(k + "/lo", NAN), M.get(k + "/hi", NAN), nd)


def _pv(M, k):
    return M[k + "/point"]


def _tab(header, rows):
    """문자열 표(빈 표 금지)."""
    assert rows, "빈 표"
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def perturbation():
    exp = PV
    M = json.loads((R / exp / "metrics.json").read_text())
    pj = pd.read_csv(R / exp / "per_joint.csv")
    JNT = [j for j in dict.fromkeys(pj.joint) if j != "all"]
    # CSV 와 metrics.json 이 같은 값인지 점검(요약은 metrics 키로 읽는다)
    for f_, keyf in ((pj, lambda r: f"pj/{r.domain}/{r.stat}/{r.joint}"),
                     (pd.read_csv(R / exp / "sample_count.csv"), lambda r: f"count/{r.domain}/{r.output}/{r.S}/{r.stat}")):
        for r in f_.itertuples():
            assert abs(M[keyf(r) + "/point"] - r.point) < 1e-9 * max(1, abs(r.point)), keyf(r)
    P = lambda dom, st, j="all", nd=3: _pk(M, f"pj/{dom}/{st}/{j}", nd)  # noqa: E731
    nw = {d: int(M[f"n_workers/{d}"]) for d in DOMS}
    nim = {d: int(M[f"n_images/{d}"]) for d in DOMS}

    # ---- 1. 관절별 실측 위치 오차
    t1 = _tab(["joint", "n", "중앙값", "68 백분위", "95 백분위", "x 표준편차", "y 표준편차"],
              [[j, f"{M[f'pj/all/n/{j}/point']:.0f}", *[P("all", s, j) for s in ("med", "p68", "p95", "sd_x", "sd_y")]] for j in JNT])
    DH = ["joint"] + [PD_KO[d] for d in PD]
    t1m = _tab(DH, [[j, *[P(d, "med", j) for d in PD]] for j in JNT])
    t1p = _tab(DH, [[j, *[P(d, "p95", j) for d in PD]] for j in JNT])
    # ---- 2. 가정 섭동과의 비
    t2 = _tab(["joint", "가정 sd 중앙값(sig_med)", "가정 sd rms(sig_rms)", "실측 강건 sd x(rsd_x)", "실측 강건 sd y(rsd_y)", "비(ratio)", "비(ratio_rms)"],
              [[j, *[P("all", s, j, 4) for s in ("sig_med", "sig_rms", "rsd_x", "rsd_y")], P("all", "ratio", j, 2), P("all", "ratio_rms", j, 2)] for j in JNT])
    t2d = _tab(DH, [[j, *[P(d, "ratio", j, 2) for d in PD]] for j in JNT])
    cat = lambda x: "2 초과" if x > 2 else ("0.5 미만" if x < 0.5 else "0.5~2")  # noqa: E731
    t2g = _tab(["영역", "관절 비의 중앙값(ratio_median)", "구간 해석 규칙의 칸", "0.5 미만 관절 수", "0.5~2 관절 수", "2 초과 관절 수", "관절 수"],
               [[PD_KO[d], P(d, "ratio_median", nd=2), cat(_pv(M, f"pj/{d}/ratio_median/all")), P(d, "n_ratio_lt05", nd=0), P(d, "n_ratio_05_2", nd=0),
                 P(d, "n_ratio_gt2", nd=0), P(d, "n_joints", nd=0)] for d in PD])
    mid = [j for j in JNT if 0.5 <= _pv(M, f"pj/all/ratio/{j}") <= 2]
    lt = [j for j in JNT if _pv(M, f"pj/all/ratio/{j}") < 0.5]
    # ---- 3. 분포 모양
    t3 = _tab(["joint", "초과 첨도 x", "초과 첨도 y", "초과 첨도(평균)", "등방성 sd_x/sd_y", "등방성 강건", "신뢰도-오차 순위 상관"],
              [[j, *[P("all", s, j, 2) for s in ("kurt_x", "kurt_y", "kurt", "iso", "iso_robust", "rho")]] for j in JNT])
    t3g = _tab(["영역", "초과 첨도 1 초과 관절 수", "관절 수", "순위 상관 절댓값의 관절 중앙값", "0.3 미만인가", "구간이 0.3 을 포함하는가"],
               [[PD_KO[d], P(d, "n_kurt_gt1", nd=0), P(d, "n_joints", nd=0), P(d, "rho_abs_median"),
                 "예" if _pv(M, f"pj/{d}/rho_abs_median/all") < 0.3 else "아니오",
                 "해당 없음(구간 없음)" if d == "all" else ("예" if M[f"pj/{d}/rho_abs_median/all/lo"] <= 0.3 <= M[f"pj/{d}/rho_abs_median/all/hi"] else "아니오")] for d in PD])
    kmaj = all(_pv(M, f"pj/{d}/n_kurt_gt1/all") > _pv(M, f"pj/{d}/n_joints/all") / 2 for d in PD)
    rho_lt = [PD_KO[d] for d in PD if _pv(M, f"pj/{d}/rho_abs_median/all") < 0.3]
    rho_ge = [PD_KO[d] for d in PD if _pv(M, f"pj/{d}/rho_abs_median/all") >= 0.3]
    # ---- 4. OKS
    t4 = _tab(["영역", "OKS 상수와 관절별 실측 오차 중앙값의 순위 상관(oks_rho_med)", "OKS 상수와 결합 강건 sd 의 순위 상관(oks_rho_rsd)"],
              [[PD_KO[d], P(d, "oks_rho_med"), P(d, "oks_rho_rsd")] for d in PD])
    # ---- 5. 각도
    A = lambda dom, kind, ang, a, nd=2: _pk(M, f"angle/{dom}/{kind}/{ang}/{a}", nd)  # noqa: E731
    t5 = {st: _tab(["angle"] + [f"sigma {s}" for s in SIGS], [[a, *[A("all", "pert", a, f"sigma{s}/{st}") for s in SIGS]] for a in ANGS]) for st in ("median", "rms", "p95")}
    t5d = _tab(["angle", "n", "중앙값", "68 백분위", "95 백분위", "평균"],
               [[a, f"{M[f'angle/all/diff/{a}/n/point']:.0f}", *[A("all", "diff", a, s) for s in ("median", "p68", "p95", "mean")]] for a in ANGS])
    t5x = _tab(["angle", "섭동 sigma 0.04 의 각도 표준편차 중앙값(도)", "섭동 sigma 0.04 의 각도 표준편차 95 백분위(도)", "예측 대 주석 각도 차 중앙값(도)", "예측 대 주석 각도 차 68 백분위(도)", "예측 대 주석 각도 차 95 백분위(도)"],
               [[a, A("all", "pert", a, "sigma0.04/median"), A("all", "pert", a, "sigma0.04/p95"), A("all", "diff", a, "median"), A("all", "diff", a, "p68"), A("all", "diff", a, "p95")] for a in ANGS])
    t5dm = _tab(["angle"] + [PD_KO[d] for d in PD], [[a, *[A(d, "diff", a, "median") for d in PD]] for a in ANGS])
    pmax = max(ANGS, key=lambda a: _pv(M, f"angle/all/diff/{a}/p95"))
    # ---- 6. 표본 수
    C = lambda dom, o, S, st, nd=4: _pk(M, f"count/{dom}/{o}/S{S}/{st}", nd)  # noqa: E731
    t6 = _tab(["output", "S", "표준오차 평균(se_mean)", "표준오차 95 백분위(se_p95)", "독립 시드 간 최빈 수준 불일치율(disagree)", "최빈 확률 차 평균(dp_mean)", "최빈 확률 차 95 백분위(dp_p95)"],
              [[o, S, *[C("all", o, S, st) for st in ("se_mean", "se_p95", "disagree", "dp_mean", "dp_p95")]] for o in POUT for S in SCNT])
    t6d = _tab(["output"] + [PD_KO[d] for d in PD], [[o, *[C(d, o, 128, "disagree") for d in PD]] for o in POUT])
    t6r = _tab(["output(행동 수준이 규칙 대상)", "S=128 불일치율(전체)", "0.02 이하인가", "영역별 점추정이 모두 0.02 이하인가", "영역별 상한(작업자 최대 또는 구간 상한)이 0.02 초과인 영역"],
               [[o, C("all", o, 128, "disagree"), "예" if _pv(M, f"count/all/{o}/S128/disagree") <= 0.02 else "아니오",
                 "예" if all(_pv(M, f"count/{d}/{o}/S128/disagree") <= 0.02 for d in DOMS) else "아니오",
                 ", ".join(DOM_KO[d] for d in DOMS if M[f"count/{d}/{o}/S128/disagree/hi"] > 0.02) or "없음"] for o in POUT])
    # ---- 7. 출력 민감도
    Sn = lambda dom, m, s, S, q, nd=3: _pk(M, f"sens/{dom}/{m}/sigma{s}/S{S}/{q}", nd)  # noqa: E731
    t7 = {q: _tab(["method", "sigma"] + [f"S {S}" for S in SCNT], [[m, s, *[Sn("all", m, s, S, q) for S in SCNT]] for m in PMETH for s in SIGS]) for q in ("keep_mode", "keep_share", "acc_mode")}
    t7a = _tab(["method", "결정론 행동 수준의 주석 참조 대비 정확도(acc_det)"], [[m, _pk(M, f"sens/all/{m}/deterministic/acc_det")] for m in PMETH])
    t7d = _tab(["method", "지표"] + [PD_KO[d] for d in PD], [[m, q, *[Sn(d, m, "0.04", 128, q) for d in PD]] for m in PMETH for q in ("keep_mode", "keep_share", "acc_mode")])
    # ---- 8. xyz 대 xy
    Z = lambda dom, *p, nd=3: _pk(M, "/".join(("zfx", dom) + p), nd)  # noqa: E731
    t8z = _tab(["통계"] + [PD_KO[d] for d in PD], [[st, *[Z(d, "z_range_over_xy_range", st, nd=4) for d in PD]] for st in ("median", "p5", "p95")])
    t8a = {st: _tab(["angle"] + [f"sigma {s}: xyz / xy" for s in SIGS],
                    [[a, *[f"{M[f'zfx/all/angle_sd/xyz/{a}/sigma{s}/{st}/point']:.3f} / {M[f'zfx/all/angle_sd/xy/{a}/sigma{s}/{st}/point']:.3f}" for s in SIGS]] for a in ANGS])
           for st in ("median", "rms", "p95")}
    t8l = _tab(["method", "sigma", "keep_mode xyz", "keep_mode xy", "차이(xyz - xy)", "keep_share xyz", "keep_share xy", "차이(xyz - xy)", "acc_mode xyz", "acc_mode xy"],
               [[m, s, Z("all", "level", "xyz", m, f"sigma{s}", "keep_mode", nd=4), Z("all", "level", "xy", m, f"sigma{s}", "keep_mode", nd=4), Z("all", "level_diff_xyz_minus_xy", m, f"sigma{s}", "keep_mode", nd=4),
                 Z("all", "level", "xyz", m, f"sigma{s}", "keep_share", nd=4), Z("all", "level", "xy", m, f"sigma{s}", "keep_share", nd=4), Z("all", "level_diff_xyz_minus_xy", m, f"sigma{s}", "keep_share", nd=4),
                 Z("all", "level", "xyz", m, f"sigma{s}", "acc_mode", nd=4), Z("all", "level", "xy", m, f"sigma{s}", "acc_mode", nd=4)] for m in PMETH for s in SIGS])
    zdiffs = [(d, m, q, M[f"zfx/{d}/level_diff_xyz_minus_xy/{m}/sigma0.04/{q}/point"]) for d in PD for m in PMETH for q in ("keep_mode", "keep_share")]
    zmax = max(abs(x[3]) for x in zdiffs)
    zall = max(abs(x[3]) for x in zdiffs if x[0] == "all")
    z_over = [x for x in zdiffs if abs(x[3]) > 0.01]
    zbad = [x for x in zdiffs if x[3] != 0]
    nz_angle = [a for a in ANGS if any(M[f"zfx/all/angle_sd/xyz/{a}/sigma{s}/{st}/point"] != M[f"zfx/all/angle_sd/xy/{a}/sigma{s}/{st}/point"] for s in SIGS for st in ("median", "rms", "p95"))]
    # ---- 본문
    hip = [j for j in mid]
    r_all = {d: _pv(M, f"pj/{d}/ratio_median/all") for d in PD}
    assert len({cat(x) for x in r_all.values()}) == 1, r_all   # 본문 문장이 네 칸이 같은 칸이라고 쓴다
    out = {d: pj[(pj.domain == d) & pj.lo.notna() & ((pj.point < pj.lo - 1e-9) | (pj.point > pj.hi + 1e-9))] for d in DOMS}
    tot = {d: int(pj[(pj.domain == d) & pj.lo.notna()].shape[0]) for d in DOMS}
    out_rm = [PD_KO[d] for d in DOMS if ((out[d].stat == 'ratio_median') & (out[d].joint == 'all')).any()]
    dis_txt = "; ".join(f"{o} {C('all', o, 128, 'disagree')}" for o in POUT[:3])
    s = head2(exp, "C5", "E3 (섭동 모형의 서술적 검증)", "cohort_pose_cache, aihub_busan_annotations 5,889장",
              "결과를 본 뒤 설계한 사후 서술 분석(판정 없음). 본 결과(xyz, sigma 0.04, S 128)를 이 분석으로 바꾸지 않는다")
    s += f"""
## 1. 범위와 정의
- 이 실험은 1차·2차 결과를 이미 본 뒤에 설계한 **사후·서술적 분석**이다. 통과/미달 판정이 없고, 사전에 정한 해석 규칙(targets.md 의 "섭동 모형 검증"과 "추가 항목 8")을 기계적으로 적용한 결과만 쓴다. sigma_base 와 S 는 이 자료에 맞춰 다시 고르지 않았고, 본 결과(xyz, sigma 0.04, S 128)는 이 분석으로 바뀌지 않는다.
- 대상: 정답 관절점이 있는 5,889장 중 잔차를 쓸 수 있는 {M['pj/n_images_with_residual']:.0f}장(예측과 주석이 모두 있는 관절만). 제조 {nim['manufacturing']}장({nw['manufacturing']}명), 요양 {nim['eldercare']}장({nw['eldercare']}명), 물류 {nim['logistics']}장({nw['logistics']}명). 단일 코호트이고 작업자 간 교환 가능성을 가정한 평균적 결과다.
- 구간: 제조는 작업자 단위 부트스트랩 95% 구간(2,000회, 시드 0). 요양(4명)과 물류(6명)는 작업자가 적어 구간 추정을 하지 않고 작업자별 값의 [최소, 최대]를 싣는다. 전체(all)는 점추정만 있다.
- **오차의 정의**: 예측 관절점과 사람이 표시한 관절점의 차이를, 골반을 원점으로 맞추고 각자의 몸통 길이(목-골반)로 나눈 값(골반 기준 상대 오차)으로 구한 뒤, 파이프라인이 섭동에 쓰는 기준 길이(ref_length = 예측 자세의 어깨 너비, 엉덩이 너비, 몸통 길이 중 최댓값) 단위로 환산했다. 관절별 중앙값, 68·95 백분위는 x, y 차이의 유클리드 거리에 대한 값이다.
- **가정 섭동**: 파이프라인은 관절 i 의 x, y, z 열에 표준편차 sigma_fraction({M['sigma_fraction_pipeline']:.2f}) × ref_length × (1 - 신뢰도_i) 인 정규 잡음을 독립으로 더한다. 기준 길이 단위로는 sigma_fraction × (1 - 신뢰도_i)이다(sig_med: 같은 관절의 중앙값, sig_rms: rms).
- **비(ratio)**: 분자는 관절별 실측 오차의 강건 표준편차(IQR / 1.349)를 x, y 두 축에서 구해 sqrt((rsd_x^2 + rsd_y^2) / 2)로 합친 값, 분모는 같은 관절·같은 이미지들의 가정 표준편차 중앙값(sig_med)이다. ratio_rms 는 분모를 rms 로 바꾼 값이다. 초과 첨도는 m4 / m2^2 - 3(x, y 평균), 등방성은 sd_x / sd_y 와 강건 표준편차 비, 순위 상관은 신뢰도와 오차 거리의 Spearman 이다.
- **각도 표준편차(항목 5)**: 예측 자세에서 S=128개를 섭동해 각도를 다시 재고, 섭동 안 한 각도 둘레에서 -180~180도로 접어(원형 처리) 표준편차를 낸 뒤 이미지에 대해 요약한다(시드 0). 각도 차는 예측 각도와 주석 각도의 차를 -180~180도로 접은 절댓값이다. 관절각 8개(좌우 포함)를 쓴다.
- **표본 수(항목 6, 7)**: 이미지마다 S_max=512개를 한 번 추출하고 S=32, 64, 128, 256, 512 는 그 앞부분을 쓴다(**중첩 설계**: S 사이의 값은 독립 추출이 아니다). 독립 시드 비교는 시드 0 과 시드 1(sigma 0.04)의 같은 앞부분끼리다. 표준오차는 sqrt(p(1 - p) / S)의 이미지 평균이다.
- **항목 8**: z 열을 섭동 뒤 원래 값으로 되돌린 변형(xy)과 파이프라인 그대로(xyz)를 같은 난수로 비교한다. 항목 8 의 행동 수준은 S=128 만 있다.
- 문헌 맥락(조사 파일 `paper/references/lit_domain_standards_20261114.md`): 몸 크기로 정규화한 관절별 위치 오차(COCO OKS 의 관절별 상수는 pycocotools `cocoeval.py` 코드 직접 확인, PCK)와 관절각 오차(도), 몬테카를로 표준오차(Morris 2019 의 sqrt(p(1 - p) / n), 전문), 행동 수준 유지율(조사 B-1, 초록만 읽음)이 이 분야의 통상 평가 요소다. 관절각 오차의 통상 규모로 확인된 값은 영상 기반 RMSE 4.77도(0.95~15.66도, Li 2020, 일부 전문)이고, B-1 의 섭동 격자는 2, 5, 8, 12도(초록)다. 신뢰도 비례 섭동과 등방 정규 섭동을 비교한 논문은 찾지 못했다(조사 파일). 아래 수치는 이 문헌값과 같은 척도가 아니다(문헌은 별도 정의).
{run_hist_pv(exp)}

## 2. 기계적 결과
### 2.1 관절별 실측 위치 오차(측정 1, 전체)
기준 길이 단위. n 은 그 관절이 주석된 이미지 수다.
{t1}

영역별 중앙값(제조 [95% 구간], 요양·물류 [작업자 최소, 최대]):
{t1m}

영역별 95 백분위:
{t1p}

### 2.2 가정 섭동 대비 비(측정 2)
{t2}

영역별 비(제조 [95% 구간], 요양·물류 [작업자 최소, 최대]):
{t2d}

영역별 관절 요약:
{t2g}

- 비가 0.5~2 인 관절(전체): {', '.join(mid) or '없음'}. 0.5 미만인 관절(전체): {', '.join(lt) or '없음'}.

### 2.3 분포 모양(측정 3, 전체)
{t3}

영역별 요약:
{t3g}

### 2.4 COCO OKS 관절별 상수와의 순위 일치(측정 4)
OKS 상수가 있는 관절 15개(COCO 17점 중 이 자료에 있는 것, 큰발가락 2개는 대응 상수가 없어 제외)에 대한 Spearman 이다. 척도(COCO 는 사람 면적의 제곱근, 여기는 기준 길이)가 달라 절대값은 비교하지 않는다. 해석 규칙은 없다.
{t4}

### 2.5 관절각 단위 환산(측정 5, 전체)
섭동 크기(sigma)별 각도 표준편차(도, 이미지 중앙값):
{t5['median']}

같은 표준편차의 이미지 rms:
{t5['rms']}

같은 표준편차의 이미지 95 백분위:
{t5['p95']}

예측 대 주석의 각도 차(도, 절댓값, 전체):
{t5d}

영역별 각도 차 중앙값(제조 [95% 구간], 요양·물류 [작업자 최소, 최대]):
{t5dm}

### 2.6 표본 수(측정 6, 전체)
sigma 0.04, 시드 0 의 표준오차와 시드 0 대 시드 1 의 최빈 수준(OWAS, RULA, REBA 는 행동 수준 / rula_grand, reba_grand 는 종합 점수) 불일치율.
{t6}

S=128 의 불일치율, 영역별:
{t6d}

### 2.7 의미 있는 출력의 민감도(측정 7, 전체)
keep_mode: 결정론 행동 수준이 최빈 수준과 같은 이미지의 비율. keep_share: 표본 중 결정론 행동 수준과 같은 비율의 평균. acc_mode: 최빈 수준이 주석 참조 행동 수준과 같은 비율. {REF_ANN}

keep_mode:
{t7['keep_mode']}

keep_share:
{t7['keep_share']}

acc_mode:
{t7['acc_mode']}

결정론 행동 수준의 주석 참조 대비 정확도:
{t7a}

sigma 0.04, S=128 의 영역별 값(제조 [95% 구간], 요양·물류 [작업자 최소, 최대]):
{t7d}

### 2.8 xyz 대 xy(측정 8)
예측 자세의 z 범위 / (x 범위와 y 범위 중 큰 값), 이미지 통계:
{t8z}

각도 표준편차(도, 전체)의 xyz(파이프라인 그대로) 대 xy(z 를 원래 값으로 되돌림), 이미지 중앙값:
{t8a['median']}

이미지 rms:
{t8a['rms']}

이미지 95 백분위:
{t8a['p95']}

행동 수준(전체, S=128), 유지율과 정확도:
{t8l}

## 3. 해석 규칙의 기계적 적용 결과
규칙은 실행 전에 targets.md 에 적어 둔 것이며 아래는 그 적용 결과만이다.
1. **비의 관절 중앙값**: 전체 {r_all['all']:.2f}, 제조 {r_all['manufacturing']:.2f}, 요양 {r_all['eldercare']:.2f}, 물류 {r_all['logistics']:.2f}로 네 칸 모두 "2 초과"에 해당한다(규칙 문구: "가정이 작다"). 영역별 관절 수는 2.2 절의 표(전체: 0.5 미만 {_pv(M, 'pj/all/n_ratio_lt05/all'):.0f}, 0.5~2 {_pv(M, 'pj/all/n_ratio_05_2/all'):.0f}, 2 초과 {_pv(M, 'pj/all/n_ratio_gt2/all'):.0f}). 0.5~2 인 관절은 {', '.join(hip) or '없음'}이고 0.5 미만인 관절은 {len(lt)}개다. {'방향이 갈리는 관절(0.5 미만)은 없다.' if not lt else '0.5 미만인 관절이 있어 방향이 갈린다.'}
2. **초과 첨도**: 1 초과 관절이 {', '.join(f'{PD_KO[d]} {_pv(M, f"pj/{d}/n_kurt_gt1/all"):.0f}/{_pv(M, f"pj/{d}/n_joints/all"):.0f}' for d in PD)}로 {'네 칸 모두 과반이다(규칙 문구: "정규보다 꼬리가 두껍다")' if kmaj else '과반이 아닌 칸이 있다'}.
3. **신뢰도-오차 순위 상관의 절댓값(관절 중앙값)**: 0.3 미만인 칸은 {', '.join(rho_lt) or '없음'}, 0.3 이상인 칸은 {', '.join(rho_ge) or '없음'}이다. 영역별로 갈린다. 0.3 미만인 칸에는 규칙 문구("신뢰도에 비례해 섭동을 주는 설계는 오차를 약하게만 반영한다")가 적용되고, 0.3 이상인 칸에는 적용되지 않는다. 제조, 요양, 물류의 구간은 0.3 을 걸치는지 여부가 2.3 절의 표에 있다.
4. **S=128 의 독립 시드 간 최빈 수준 불일치율**(행동 수준, 방법별): {dis_txt}. 세 방법 모두 0.02 {'이하' if all(_pv(M, f'count/all/{o}/S128/disagree') <= 0.02 for o in POUT[:3]) else '이하가 아닌 방법이 있음'}이므로 규칙 문구는 "128 은 최빈 수준에 충분"이다(전체 기준). 영역별로는 2.6 절 끝의 표(작업자 최대 또는 구간 상한이 0.02 를 넘는 영역이 있다). 규칙의 대상이 아닌 종합 점수는 reba_grand 가 {C('all', 'reba_grand', 128, 'disagree', 6)}로 0.02 {'를 넘는다' if _pv(M, 'count/all/reba_grand/S128/disagree') > 0.02 else '이하다'}. 표준오차는 2.6 절의 수치 그대로다.
5. **z 차이**: sigma 0.04 에서 행동 수준 유지율(keep_mode, keep_share)의 xyz 대 xy 차이의 절댓값 최댓값은 전체 {zall:.4f}, 영역을 포함한 모든 칸에서 {zmax:.4f}이다. 1 %p(0.01)를 넘는 칸은 {len(z_over)}개이고 0 이 아닌 칸은 {len(zbad)}개다(전체·영역 {len(zdiffs)}칸 중). 규칙 문구는 {'"z 섭동이 점수 분포에 실질적으로 기여한다"' if z_over else '"영향이 작다"(수치는 보충자료)'}이다. 본 결과(xyz)는 xy 로 바꾸지 않는다.

## 4. 관찰된 이상 징후
- **엉덩이 관절의 비**: 비가 0.5~2 인 관절은 {', '.join(hip) or '없음'}뿐이고(전체 {', '.join(P('all', 'ratio', j, 2) for j in hip)}), 나머지는 2 를 넘는다. 오차가 골반을 원점으로 한 상대 오차이므로 엉덩이 관절(골반에 가까운 점)의 오차가 작게 잡히는 정의의 영향일 수 있다. 이 실험에서 그 원인을 검정하지 않았으므로 단정하지 않는다.
- **관절점 수준과 각도 수준이 주는 그림의 차이**: 관절점 수준에서 비의 관절 중앙값은 전체 {P('all', 'ratio_median', nd=2)}(실측 오차의 강건 표준편차가 가정의 약 5배 규모)인 반면, 각도 수준에서는 sigma 0.04 의 섭동이 만드는 각도 표준편차 중앙값이 몸통 {A('all', 'pert', 'trunk', 'sigma0.04/median')}도, 목 {A('all', 'pert', 'neck', 'sigma0.04/median')}도, 아래팔(왼쪽) {A('all', 'pert', 'forearm_l', 'sigma0.04/median')}도, 무릎(왼쪽) {A('all', 'pert', 'knee_l', 'sigma0.04/median')}도이고, 예측 대 주석 각도 차의 중앙값은 각각 {A('all', 'diff', 'trunk', 'median')}도, {A('all', 'diff', 'neck', 'median')}도, {A('all', 'diff', 'forearm_l', 'median')}도, {A('all', 'diff', 'knee_l', 'median')}도다(2.5 절의 표). 두 값은 정의가 다르다(섭동의 표준편차 대 한 이미지에서 예측과 주석의 각도 차 절댓값). 두 수준의 차이를 만든 원인은 분석하지 않았다.
- **각도 차의 95 백분위가 매우 크다**: 전체 95 백분위가 {', '.join(f"{a} {A('all', 'diff', a, 'p95', 1)}도" for a in ANGS)}이고 가장 큰 것은 {pmax} 이다. 평균이 중앙값보다 훨씬 크다(2.5 절의 표). 좌우 뒤바뀜이나 얼굴 방향 판정 같은 원인이 가능하지만 이 실험에서 검정하지 않았다.
- **점추정이 구간 밖인 칸**: per_joint.csv 의 구간이 있는 통계 행 중 점추정(영역 전체 이미지를 모은 값)이 제조 부트스트랩 구간 또는 요양·물류 작업자 [최소, 최대] 밖에 있는 행은 제조 {len(out['manufacturing'])} / {tot['manufacturing']}, 요양 {len(out['eldercare'])} / {tot['eldercare']}, 물류 {len(out['logistics'])} / {tot['logistics']}이다. 비의 관절 중앙값에서 점추정이 구간 밖인 영역: {', '.join(out_rm) or '없음'}(2.2 절의 영역별 관절 요약). 요양·물류의 작업자 범위는 작업자 4명, 6명의 값이라 풀링한 값과 같은 통계가 아니다. 원인은 분석하지 않았다.
- **xyz 대 xy**: 행동 수준 유지율의 차이는 전체 {len(zdiffs)}칸 중 0 이 아닌 것이 {len(zbad)}개다. 각도 표준편차는 {', '.join(nz_angle) or '없음'}에서 xyz 와 xy 가 다르고 나머지 각도는 모든 sigma 에서 같다. 예측 자세의 z 범위는 x, y 범위의 중앙값 {Z('all', 'z_range_over_xy_range', 'median', nd=4)}배다. 이 실험에서 z 열이 점수 계산에 쓰이는 경로는 확인하지 않았다.

## 5. 한계
- 사후 분석이다(결과를 본 뒤 설계). 판정이 없고 sigma_base 와 S 의 재선택도 없다.
- 오차는 골반 기준 상대 오차(몸통 길이로 나눈 값)이며, 엉덩이 관절의 비가 작게 나오는 것은 이 정의의 영향일 수 있다(위, 단정하지 않음).
- 정답은 사람이 표시한 2D 주석이다. 주석 자체의 오차가 실측 오차에 들어 있고(분리하지 않음), 시점 오류나 3D 오차를 대변하지 않는다. 독립 기준은 전문가 판단도 실제 인간공학 위험도도 아니다.
- 표본 수 항목(S 32~512)은 한 번 추출한 512개의 앞부분을 쓰는 중첩 설계라 S 사이의 값이 독립이 아니다. 독립 시드는 2개(0, 1)뿐이라 시드 간 변동의 분포는 추정하지 않았다.
- 단일 코호트이고 요양(4명)과 물류(6명)는 작업자 수가 적어 구간 대신 작업자별 값의 범위를 실었다.
- 각도 차의 큰 꼬리와 관절점 수준의 비는 원인을 분석하지 않았다.
- 항목 8 의 행동 수준은 S=128 만 있고, 난수는 xyz 와 xy 가 같다.

## 6. 재현
{run_info2(exp)}
- 입력: run.json 의 data({', '.join(d['name'] for d in json.loads((R / exp / 'run.json').read_text()).get('data', []))}).
- metrics.json 키 {len(M)}개와 per_joint.csv, sample_count.csv 의 값이 같음을 이 요약을 만들 때 대조했다.
"""
    (R / exp / "summary.md").write_text(s)


def run_hist_pv(exp):
    """실행 정보 줄(RUN_PREFIX 로 시작): 이 실험은 심사 봉인 대상이 아니라 실행 이력만 적는다."""
    r = json.loads((R / exp / "run.json").read_text())
    return (f"- 실행 이력: 정식 실행 1회(종료 코드 {r['exit_code']}). run.json 의 git_dirty({r.get('git_dirty')})는 결과 파일이 쓰인 뒤 기록되므로 코드 동일성은 "
            f"code_hash({r['code_hash'][:12]})로 확인한다.")


def _ek(M, k, nd=3):
    return ci(M[k], M.get(k + "_lo", NAN), M.get(k + "_hi", NAN), nd)


def expert():
    exp = EA_
    M = json.loads((R / exp / "metrics.json").read_text())
    ag = pd.read_csv(R / exp / "agreement.csv")
    pi_ = pd.read_csv(R / exp / "per_image.csv")
    SET = ("submitted", "cohort", "taskload")
    SET_KO = {"submitted": "submitted(제출 당시)", "cohort": "cohort(코호트 본분석 설정)", "taskload": "taskload(전문가 비교 설정)"}
    MET = ("OWAS", "RULA", "REBA")
    n = int(M["n_images"])
    # CSV 와 metrics 대조, per_image 로 정확 일치 장수와 방향 재계산
    for r in ag.to_dict("records"):
        for c, v in r.items():
            if c not in ("setting", "method") and not pd.isna(v):
                assert abs(M[f"{r['setting']}/{r['method']}/{c}"] - v) < 1e-9, (r["setting"], r["method"], c)
    dirs = {}
    for (st, me), g in pi_.groupby(["setting", "method"]):
        assert int((g.system == g.expert).sum()) == M[f"{st}/{me}/exact_n"], (st, me)
        dirs[(st, me)] = (int((g.system < g.expert).sum()), int((g.system == g.expert).sum()), int((g.system > g.expert).sum()))
    K = lambda st, me, c, nd=3: _ek(M, f"{st}/{me}/{c}", nd)  # noqa: E731
    E1 = _tab(["설정", "method", "n", "정확 일치 장수", "정확 일치 비율 [95% 구간]", "한 단계 이내 비율 [95% 구간]"],
              [[SET_KO[st], me, f"{M[f'{st}/{me}/n']:.0f}", f"{M[f'{st}/{me}/exact_n']:.0f} / {M[f'{st}/{me}/n']:.0f}", K(st, me, "exact_rate"), K(st, me, "within1_rate")] for st in SET for me in MET])
    E2 = _tab(["설정", "method", "선형 가중 kappa [95% 구간]", "이차 가중 kappa [95% 구간]", "평균 부호 차이(시스템 - 전문가) [95% 구간]"],
              [[SET_KO[st], me, K(st, me, "kappa_linear"), K(st, me, "kappa_quadratic"), K(st, me, "mean_signed_diff")] for st in SET for me in MET])
    E3 = _tab(["설정", "method", "시스템이 전문가보다 낮음", "같음", "시스템이 전문가보다 높음"],
              [[SET_KO[st], me, *[f"{x} / {n}" for x in dirs[(st, me)]]] for st in SET for me in MET])
    # 원고 대조(submitted, taskload): 키가 있는 것만
    mrows = []
    for st in ("submitted", "taskload"):
        for me in MET:
            k = f"{st}/{me}"
            mrows.append([SET_KO[st], me, f"{M[k + '/manuscript_exact_n']:.0f}", f"{M[k + '/exact_n']:.0f}", "같음" if M[k + "/same_exact_as_manuscript"] else "다름",
                          f"{M[k + '/manuscript_mean_abs_diff']:.2f}", f"{M[k + '/mean_abs_diff']:.4f}", "같음(소수 둘째 자리)" if M[k + "/same_mean_abs_diff_as_manuscript"] else "다름"])
    E4 = _tab(["설정", "method", "원고 표 10 정확 일치", "재계산 정확 일치", "일치 장수", "원고 표 10 평균 절대 차이", "재계산 평균 절대 차이", "평균 절대 차이"], mrows)
    # 본문 "접촉 규칙 없이 12, 9, 4 (OWAS, REBA, RULA)" 대 cohort. 원고 값은 원고에서 읽은 것이며 metrics 에 없다.
    MS_TXT = {"OWAS": 12, "REBA": 9, "RULA": 4}
    E5 = _tab(["method", "원고 본문(접촉 규칙 없는 설정) 정확 일치", "cohort 재계산 정확 일치", "일치 여부", "원고 본문의 '시스템이 전문가보다 낮음' 장수", "cohort 재계산"],
              [[me, MS_TXT[me], f"{M[f'cohort/{me}/exact_n']:.0f}", "같음" if MS_TXT[me] == M[f"cohort/{me}/exact_n"] else "다름",
                {"RULA": 17}.get(me, "원고에 없음"), f"{dirs[('cohort', me)][0]}"] for me in ("OWAS", "REBA", "RULA")])
    # 해석 규칙
    cells = [(st, me, c) for st in SET for me in MET for c in ("kappa_linear", "kappa_quadratic")]
    cz = {c: [(st, me) for st in SET for me in MET if M[f"{st}/{me}/{c}_lo"] <= 0 <= M[f"{st}/{me}/{c}_hi"]] for c in ("kappa_linear", "kappa_quadratic")}
    czs = {c: [(st, me) for st in SET for me in MET if M[f"{st}/{me}/{c}_lo"] <= 0] for c in ("kappa_linear", "kappa_quadratic")}
    kl = [M[f"{st}/{me}/kappa_linear"] for st in SET for me in MET]
    low = [(me, M[f"cohort/{me}/exact_n"], M[f"taskload/{me}/exact_n"]) for me in MET if M[f"cohort/{me}/exact_n"] < M[f"taskload/{me}/exact_n"]]
    same = [me for me in MET if M[f"cohort/{me}/exact_n"] == M[f"taskload/{me}/exact_n"]]
    klow = [me for me in MET if M[f"cohort/{me}/kappa_linear"] < M[f"taskload/{me}/kappa_linear"]]
    E6 = _tab(["method", "cohort 정확 일치", "taskload 정확 일치", "cohort 선형 kappa [95% 구간]", "taskload 선형 kappa [95% 구간]", "cohort 이차 kappa [95% 구간]", "taskload 이차 kappa [95% 구간]"],
              [[me, f"{M[f'cohort/{me}/exact_n']:.0f}", f"{M[f'taskload/{me}/exact_n']:.0f}", K("cohort", me, "kappa_linear"), K("taskload", me, "kappa_linear"),
                K("cohort", me, "kappa_quadratic"), K("taskload", me, "kappa_quadratic")] for me in MET])
    neg = [(st, me) for st in SET for me in MET if M[f"{st}/{me}/mean_signed_diff"] < 0]
    # 이전 각도 코드 단계표(실행 기록 없는 참고 파일)와 현재 cohort 의 이미지별 대조
    PREV = R.parents[2] / "ace_pose/_runs/expert_pilot/stages_previous_angle_code.csv"
    prev = pd.read_csv(PREV)
    pdiff, prow = {}, []
    for me in MET:
        a = prev[prev.standard == me].set_index("id")
        c = pi_[(pi_.setting == "cohort") & (pi_.method == me)].set_index("id")
        assert (a.expert.reindex(c.index) == c.expert).all(), me          # 전문가 열이 per_image.csv 와 같다
        old, cur = a.angles_nocontact.reindex(c.index), c.system
        pdiff[me] = [f"{i}({int(old[i])} 대 {int(cur[i])})" for i in c.index if old[i] != cur[i]]
        prow.append([me, int((old == c.expert).sum()), int((cur == c.expert).sum()), len(pdiff[me]), ", ".join(pdiff[me]) or "없음"])
    E7 = _tab(["method", "이전 각도 코드(접촉 규칙 제외) 정확 일치 [참고 파일 값]", "현재 cohort 정확 일치", "이미지별로 달라진 장수", "달라진 이미지(이전 대 현재 수준)"], prow)
    # 주변 분포(per_image.csv 집계)
    mrow = []
    for me in MET:
        for lv in LEVELS_EA[me]:
            mrow.append([me, lv, int(((pi_.setting == "cohort") & (pi_.method == me) & (pi_.expert == lv)).sum())] +
                        [int(((pi_.setting == st) & (pi_.method == me) & (pi_.system == lv)).sum()) for st in SET])
    E8 = _tab(["method", "행동 수준", "전문가(장)", "submitted 시스템(장)", "cohort 시스템(장)", "taskload 시스템(장)"], mrow)
    modal = {me: int(pi_[(pi_.setting == "cohort") & (pi_.method == me)].expert.value_counts().max()) for me in MET}
    nm = lambda ps: ", ".join(f"{st}/{me}" for st, me in ps) or "없음"  # noqa: E731
    s = head2(exp, "C1", "E1 (전문가 22장과의 표준 일치 지표)", "cohort_pose_cache, expert_pilot_22",
              "22장은 오류 수정에 쓴 개발 세트이고 평가자 1인이다. 이 값은 독립 검증이 아니다")
    s += f"""
## 1. 범위와 정의
- **22장은 오류 수정에 쓴 개발 세트이고 평가자는 1인(프로젝트 소속 인간공학 전문가)이다. 이 요약의 값은 독립 검증이 아니다.** 전문가 판단은 실제 인간공학 위험도가 아니다. 새 전문가 자료는 없고 시스템 점수만 다시 계산했다(자세는 코호트 캐시의 22장 기존 추론, 새 추론 없음).
- **설정 세 가지**:
  - `submitted`: 전문가에게 제출됐던 시스템 점수(등록 자료 그대로, 재계산 아님). 원고 표 10 의 첫 행(전문가에게 제출한 점수)에 대응한다.
  - `cohort`: 코호트 본분석 설정(`cohort_context`: 하중 0 kg, 손목 기본값 그대로). 접촉 규칙 없음. 표 10 에 대응하는 행이 없다(표 10 의 행은 8단계가 전부 누적 단계이고, 이 설정은 하중 입력을 0 kg 으로 둔 현재 코드의 값이다).
  - `taskload`: 전문가 비교 설정(`task_context`: 작업별 잠정 하중표). 표 10 의 8단계("Task-level load (provisional)")에 대응한다.
  - **taskload 의 순환성**: 작업별 하중표(`ace_pose/standards/task_context.json`, 파일 시각 2026-10-08 21:27)는 같은 날(2026-10-08, `paper/strategy-log.md` 의 "시도 S1") 전문가의 22장 채점표를 본 뒤의 진단에서 나왔다(진단 때 하위 점수를 전문가 값으로 바꿔 넣어 보았고, 전문가의 하중 판정이 작업과 일치함을 확인한 뒤 공정 단위 하중 표를 넣었다). 표의 `_doc` 은 값이 이미지별 전문가 점수에서 오지 않고 문헌 또는 전문가의 작업 단위 판단에서 왔다고 적지만, 표는 저자가 잠정 지정한 것이고 전문가가 확인하지 않았다(전문가 회신은 더 없다: `paper/decision-log.md` 2026-11-13). 따라서 cohort 에서 taskload 로 RULA 4 대 10, REBA 6 대 10 이 되는 것은 같은 22장을 본 뒤에 맞춘 입력의 값이며 **개선의 증거가 아니다**. 이 요약은 이를 "하중 입력을 바꾸면 이 22장의 일치 장수가 달라진다"는 기술로만 쓴다.
- **재현하지 못한 단계**: 표 10 의 1~7단계는 접촉 하중 규칙(이미지를 다시 추론해야 얻는 제2 인물 검출이 필요)과 폐기된 이전 각도 코드를 쓰므로 여기서 재현하지 않는다. 표 10 의 첫 행과 8단계만 대조한다.
- 전문가 종합 점수 열이 없어 **행동 수준만** 계산했다(종합 점수의 일치 지표는 없다). 행동 수준 범주 집합은 관측값이 아니라 방법별 가능한 전체로 고정했다(OWAS 1~4, RULA 1~4, REBA 0~4).
- 지표: 정확 일치 장수와 비율, 한 단계 이내 비율, 선형·이차 가중 kappa(가중 = (|i - j| / (범주 수 - 1))^지수, 지수 1·2), 부호 있는 평균 차이(시스템 - 전문가). 구간은 이미지 단위 부트스트랩 백분위 95%(2,000회, 시드 0)이며, 표본에서 kappa 가 정의되지 않으면 그 표본은 해당 지표에서 뺀다(이번 실행에서 그런 표본은 없었다: metrics.json 에 *_undefined 키 없음).
- n = {n}이고 구간이 넓다.
- 문헌 맥락(조사 파일 `paper/references/lit_review_items_20261114.md` 주제 F): 이 분야에서 자동 점수와 전문가의 일치는 가중 kappa 와 일치율(구간 포함)로 보고하는 것이 통상이다. 확인된 자동 대 전문가 kappa 범위는 대략 0.46~0.74(Li 2020 평균 0.738 전문, Kwon 2022 부위별 0.456~0.704 전문, Plantard 2017 은 Scataglini 2025 가 요약한 2차 값)이며, 이 요약의 22장은 개발 세트, 평가자 1인, 독립 검증 아님이라는 점을 같은 문장에 둔다.
{run_hist_pv(exp)}

## 2. 기계적 결과
**결론: 이 결과는 "자동 점수가 전문가와 일치한다"는 주장을 받치지 못한다. 선형 가중 kappa 점추정은 {min(kl):.3f}~{max(kl):.3f}이고 아홉 칸 중 {len(cz['kappa_linear'])}칸에서 구간이 0 을 포함한다(22장 개발 세트, 평가자 1인).** 코호트 결과를 낸 설정(cohort)의 값은 아래 cohort 행이다.

정확 일치와 한 단계 이내(행동 수준):
{E1}

kappa 와 평균 부호 차이:
{E2}

시스템과 전문가의 방향(행동 수준, 22장):
{E3}

## 3. 원고 기존 값과의 대조
원고 표 10 의 첫 행과 8단계에 적힌 값(재계산이 `MANUSCRIPT_EXACT`, `MANUSCRIPT_MAD` 로 대조):
{E4}

원고 본문("Without the contact rule, ... agreement is 12, 9, and 4 of 22 for OWAS, REBA, and RULA, with RULA below the expert in 17 of 22")과 현재 코드의 cohort 설정:
{E5}

- 정확 일치 장수가 다른 칸: {', '.join(me for me in MET if MS_TXT[me] != M[f'cohort/{me}/exact_n']) or '없음'}. 같은 칸: {', '.join(me for me in MET if MS_TXT[me] == M[f'cohort/{me}/exact_n']) or '없음'}. 표 10 의 첫 행과 8단계는 정확 일치 장수와 평균 절대 차이(소수 둘째 자리)가 6칸 모두 같다.
- 본문 문장의 값 중 다른 것은 재계산 값(실행 기록이 있는 값)을 쓰고 차이를 답변서에 밝힌다(해석 규칙). 본문의 "12, 9, 4"는 표 10 의 어느 행에도 없는 값이다(표 10 의 7단계 행은 12, 4, 8).

### 3.1 본문 REBA 9 의 출처(확인된 사실만)
- 본문의 "12, 9, 4"와 같은 값이 `ace_pose/_runs/expert_pilot/stages_previous_angle_code.csv`(파일 시각 2026-10-08 22:18)의 `angles_nocontact` 열(폐기된 이전 각도 코드, 접촉 규칙 제외)에서 나온다. `paper/strategy-log.md` "시도 S1" 도 같은 날 "최종 코드(접촉 규칙 제외)의 정확 일치는 OWAS 12, REBA 9, RULA 4/22"라고 적고, 팔 부호·거상과 목 Frankfurt 기준을 고친 직후(맥락 0) REBA 가 7 이 되었다고 적는다. 즉 9 는 그 코드 상태의 값이다. 이 파일의 값은 **실행 기록 없는 참고값**이다(`run.json` 이 없고 이 실험의 metrics 에 키가 없다).
- 현재 cohort 설정과의 차이로 코드와 기록에서 확인되는 것: 각도 코드 개정(팔 부호·거상 오판, 목 Frankfurt 기준), RULA 손목 기본값 2 에서 1 로, 앉은 자세 판정과 OWAS 다리 규칙 수정, 하중 입력(cohort 는 0 kg 으로 고정). 이 중 어느 것이 REBA 9 에서 6 으로의 변화를 만들었는지는 단계별로 되돌려 실행한 기록이 없어 **특정하지 않았다**.
- 참고 파일의 이미지별 값과 현재 cohort 를 대조하면(달라진 이미지 목록은 per_image.csv 와 위 참고 파일의 집계):
{E7}
- 위 대조는 이미지별 수준이 달라졌다는 사실까지다. 이전 코드 자체가 없어 달라진 이유를 이미지별로 재실행해 확인할 수 없다.
- **후속 조치**: 이 값은 실행 기록으로 재현되지 않으므로 원고 본문의 "12, 9, 4"는 삭제하고 cohort 재계산 값(OWAS 12, REBA 6, RULA 4)으로 바꾼다. 원고 표 10 의 1~7단계도 같은 이유로 "이전 코드 기록, 재현되지 않음"으로 표시하거나 뺀다(이 요약은 원고를 수정하지 않는다).

## 4. 해석 규칙의 기계적 적용 결과
- kappa 는 구간과 함께만 쓴다. 22장에서는 구간이 넓어 Landis-Koch 등급 이름을 붙이지 않는다.
- 선형 가중 kappa 의 구간이 0 을 포함하는 칸: {len(cz['kappa_linear'])} / 9 ({nm(cz['kappa_linear'])}). 이차 가중 kappa: {len(cz['kappa_quadratic'])} / 9 ({nm(cz['kappa_quadratic'])}). 구간 하한이 0 이하인 칸은 선형 {len(czs['kappa_linear'])}, 이차 {len(czs['kappa_quadratic'])}이다.
- 선형 가중 kappa 의 점추정은 9칸에서 {min(kl):.3f}~{max(kl):.3f}이다. 이 값을 문헌의 kappa 범위(대략 0.46~0.74)와 나란히 놓을 때는 22장이 개발 세트이고 평가자 1인이며 독립 검증이 아니라는 점을 함께 둔다.
- **문헌 kappa 와의 격차**: 원인을 분석하지 않았고 이 자료(22장, 평가자 1인, 종합 점수 없음)로는 분석할 수 없다. 문헌 값은 평가자 수, 이미지 구성, 가중 방식이 다르므로 직접 비교할 수 없다. 아래는 검정하지 않은 **가능한 요인(가설일 뿐)**이다. (a) 평가자가 1인이라 기준(전문가) 판단의 신뢰도를 모른다(평가자 내 재평가, 평가자 간 일치 없음). (b) 22장의 선정 방식: 원고는 대표 작업자의 22장(요양 10, 물류 6, 제조 6)이라고만 적고, 선정 절차의 기록은 이 저장소의 `src/select_heldout_images.py`(검증 36장용으로 22장 선정과 무관)나 `paper/decision-log.md` 에서 확인되지 않았다. (c) 행동 수준 범주가 적어 분포가 한쪽으로 치우치면 kappa 가 낮아지는 성질(주변 분포는 아래 표). (d) 영상에 없는 입력(하중, 보행, 지속·반복 활동)이 전문가 판단에 들어 있다(원고 4.9 절의 전문가 사유 서술). 네 요인 모두 이 요약에서 검정하지 않았다.

행동 수준의 주변 분포(출처: per_image.csv 집계, metrics.json 에 대응 키 없음; 전문가 열은 세 설정에서 같다):
{E8}

전문가 최빈 수준만 항상 내는 기준선의 정확 일치 장수(per_image.csv 집계): {', '.join(f'{me} {modal[me]}' for me in MET)}(22장 중).
- cohort 대 taskload(정확 일치 장수와 kappa, 구간은 2절):
{E6}

- cohort 의 정확 일치가 taskload 보다 낮은 방법: {', '.join(f'{me}({a} 대 {b})' for me, a, b in low) or '없음'}. 같은 방법: {', '.join(same) or '없음'}. 선형 kappa 점추정이 cohort 에서 낮은 방법: {', '.join(klow) or '없음'}. 규칙 문구에 따라 쓴다: **전문가와 맞춰 본 설정(taskload)은 코호트 결과를 낸 설정이 아니며, 코호트 설정(cohort)의 일치는 위 cohort 행의 값이다.**

## 5. 관찰된 이상 징후
- 평균 부호 차이가 음수(시스템이 전문가보다 낮음)인 칸: {len(neg)} / 9 ({nm(neg)}). 나머지 칸은 0 이상이다.
- 같은 설정·방법에서 선형과 이차 가중 kappa 가 크게 다른 칸이 있다(예: cohort REBA 선형 {K('cohort', 'REBA', 'kappa_linear')}, 이차 {K('cohort', 'REBA', 'kappa_quadratic')}). 원인은 분석하지 않았다.
- 본문의 REBA 값(12, 9, 4 중 9)과 현재 코드의 cohort REBA({M['cohort/REBA/exact_n']:.0f}장)가 다르다(3절). 원인은 분석하지 않았다.

## 6. 한계
- 22장은 오류를 찾아 고친 개발 세트이므로 독립 검증이 아니다. 평가자 1인이며 평가자 내 재평가(같은 평가자의 시간 간격 재채점)가 없어 기준 판단의 안정성을 모른다.
- 구간은 22장에 대한 부트스트랩이라 넓고(2절), 작업자 수준의 구조(요양 10, 물류 6, 제조 6장으로 대표 작업자를 고름)를 반영하지 않는다.
- cohort 설정의 하중은 전 이미지에서 0 kg, taskload 설정의 하중표는 저자의 잠정 지정이며 전문가가 확인하지 않았다. 두 설정의 차이는 입력 하나(하중)다(원고 표 10 아래 서술).
- 표 10 의 1~7단계와 종합 점수 일치는 계산하지 않았다.
- kappa 는 범주 집합을 가능한 전체로 고정해 계산했으므로 관측된 범주만 쓴 값과 다를 수 있다. 한 범주만 관측되는 표본이면 정의되지 않는다.

## 7. 재현
{run_info2(exp)}
- 입력: run.json 의 data({', '.join(d['name'] for d in json.loads((R / exp / 'run.json').read_text()).get('data', []))}).
- agreement.csv 와 metrics.json 의 값이 같고, per_image.csv 로 센 정확 일치 장수가 metrics 의 exact_n 과 같음을 이 요약을 만들 때 대조했다.
"""
    (R / exp / "summary.md").write_text(s)



RUN_PREFIX = ("- 재현 명령", "- 종료 코드", "- 실행 이력", "- 데이터 사용 기록")   # run_info, run_info2, run_hist 가 만드는 줄


def latest_pass(exp):
    """paper/reviews/ai-verify/*/verdict.seal.json 중 summary 가 이 실험인 가장 최근(폴더명 순) 심사가 PASS 인가."""
    seals = sorted((p.parent.name, json.loads(p.read_text())) for p in (R.parent / "paper/reviews/ai-verify").glob("*/verdict.seal.json"))
    mine = [j for _, j in seals if j.get("summary") == f"result/{exp}/summary.md"]
    return bool(mine) and mine[-1].get("verdict") == "PASS"


def body(txt):
    """실행 정보 줄을 뺀 본문."""
    return "\n".join(l for l in txt.splitlines() if not l.startswith(RUN_PREFIX))


def sealed(f, exp):
    """최신 심사가 PASS 인 요약: 실행 정보 줄을 뺀 본문이 같으면 기존 파일을 그대로 둔다(재실행이 통과 봉인을 깨지 않게).
    본문이 다르면 경고를 내고 새로 쓴다. PASS 가 아니면 그냥 새로 쓴다."""
    p = R / exp / "summary.md"
    if not (latest_pass(exp) and p.exists()):
        return f()
    old = p.read_text()
    f()
    new = p.read_text()
    if body(old) == body(new):
        p.write_text(old)
        return "sealed: 본문 동일, 기존 파일 유지"
    print(f"경고: {exp} 는 최신 심사가 PASS 인데 본문이 달라져 새로 썼다(봉인이 깨졌으니 재심사 필요)")
    return "본문이 달라져 새로 씀(봉인 확인 필요)"


EXPS = {sensitivity: "exp-r2-mc-sensitivity-v2", indep: "exp-r2-indep-calibration", fusion: "exp-r2-fusion-comparison-v2",
        assumed: "exp-r2-assumed-inputs", noise: "exp-r2-empirical-noise", selective: "exp-r2-selective",
        perturbation: "exp-r2-perturbation-validation", expert: "exp-r2-expert-agreement"}


if __name__ == "__main__":
    for f, exp in EXPS.items():
        note = sealed(f, exp)
        print("ok", f.__name__, note or "")
