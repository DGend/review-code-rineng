# PWW-DISCLOSURE: public
"""가중 kappa, 한 단계 이내 비율, 부트스트랩 결정론."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "topic/E/src"))
import exp_r2_expert_agreement as ex  # noqa: E402

L = [1, 2, 3, 4]


def test_kappa_perfect_is_one():
    a = np.array([1, 2, 3, 4, 2, 2])
    assert ex.weighted_kappa(a, a, L, 1) == 1.0 and ex.weighted_kappa(a, a, L, 2) == 1.0


def test_kappa_hand_example():
    # 표본 4장: 시스템 [1,1,2,2], 전문가 [1,2,2,2], 범주 {1,2}(k=2): 선형=이차=비가중 kappa.
    # 관측 일치 3/4, 주변: 시스템 (.5,.5), 전문가 (.25,.75) -> 우연 일치 .5*.25+.5*.75=.5, kappa=(.75-.5)/.5=.5
    s, e = np.array([1, 1, 2, 2]), np.array([1, 2, 2, 2])
    assert abs(ex.weighted_kappa(s, e, [1, 2], 1) - 0.5) < 1e-12
    assert abs(ex.weighted_kappa(s, e, [1, 2], 2) - 0.5) < 1e-12


def test_kappa_undefined_and_sklearn():
    assert np.isnan(ex.weighted_kappa(np.array([2, 2]), np.array([2, 2]), L, 1))
    try:
        from sklearn.metrics import cohen_kappa_score as ck
    except ImportError:
        return
    rng = np.random.default_rng(1)
    s, e = rng.integers(1, 5, 30), rng.integers(1, 5, 30)
    for p, w in ((1, "linear"), (2, "quadratic")):
        assert abs(ex.weighted_kappa(s, e, L, p) - ck(s, e, labels=L, weights=w)) < 1e-12


def test_within1_and_signed_diff():
    st = ex.stats(np.array([1, 3, 4, 2]), np.array([2, 1, 4, 2]), L)
    assert st["within1_rate"] == 0.75 and st["exact_rate"] == 0.5 and st["mean_signed_diff"] == 0.25


def test_bootstrap_deterministic():
    s, e = np.array([1, 2, 3, 3, 2, 1, 4, 2]), np.array([1, 2, 2, 3, 3, 1, 4, 1])
    a, b = ex.bootstrap(s, e, L, n_boot=200), ex.bootstrap(s, e, L, n_boot=200)
    assert a == b and a["exact_rate"][0] <= ex.stats(s, e, L)["exact_rate"] <= a["exact_rate"][1]
