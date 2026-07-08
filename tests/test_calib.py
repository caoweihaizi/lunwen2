"""CQR 校准单元测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calib.cqr import nonconformity_score, FixedCQR, DelayedLabelSlidingCQR
from src.calib.evaluate import picp, mpiw, evaluate_intervals


def test_nonconformity():
    # y_true 在区间内 -> s=0
    assert nonconformity_score(np.array([1.0]), np.array([3.0]), np.array([2.0]))[0] == 0
    # y_true 低于下界 -> s = low - true
    assert nonconformity_score(np.array([2.0]), np.array([3.0]), np.array([1.0]))[0] == 1.0
    # y_true 高于上界 -> s = true - up
    assert nonconformity_score(np.array([1.0]), np.array([2.0]), np.array([4.0]))[0] == 2.0


def test_fixed_cqr_coverage():
    rng = np.random.RandomState(0)
    # calib 集：真值在 [low, up] 内外各半
    n = 1000
    low = rng.normal(0, 1, n); up = low + 2
    true = low + rng.uniform(-1, 3, n)
    cqr = FixedCQR(target=0.90).fit(low, up, true)
    assert cqr.q_hat > 0
    # 校准后区间应更宽
    cl, cu = cqr.calibrate(low, up)
    assert (cu - cl >= up - low).all()


def test_delayed_label_no_future_leak():
    """t 时刻窗口不含 τ+h > t 的预测。"""
    scqr = DelayedLabelSlidingCQR(target=0.90, h=1)
    # τ=0 发出预测，真值在 τ+1=1 到达
    scqr.update(tau=0, pred_low=0, pred_up=2, y_true_arrived=1.5, m_observed=True)
    assert len(scqr.scores) == 1
    # 真值未到达（None）不进窗口
    scqr.update(tau=1, pred_low=0, pred_up=2, y_true_arrived=None, m_observed=False)
    assert len(scqr.scores) == 1
    # 永久缺失（m=False）不进窗口
    scqr.update(tau=2, pred_low=0, pred_up=2, y_true_arrived=1.0, m_observed=False)
    assert len(scqr.scores) == 1


def test_picp():
    y_low = np.array([0, 0, 0]); y_up = np.array([2, 2, 2]); y_true = np.array([1, 3, 0.5])
    assert abs(picp(y_low, y_up, y_true) - 2 / 3) < 1e-6


def test_evaluate_intervals():
    y_low = np.array([0.0]); y_up = np.array([2.0]); y_true = np.array([1.0])
    r = evaluate_intervals(y_low, y_up, y_true, target=0.9)
    assert r["PICP"] == 1.0
    assert r["MPIW"] == 2.0
    assert abs(r["Coverage_Gap"] - 0.1) < 1e-6
