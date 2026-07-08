"""延迟标签感知滑动 CQR 校准（技术大纲 §5.5）。

- 非一致性分数 s = max(y_low - y_true, y_true - y_up, 0)
- 延迟标签：时刻 t 的窗口 C_t = {τ: a_{τ+h} ≤ t 且 m_{τ+h}=1}
- 固定 CQR：用独立 calib 集算 q_hat
- 滑动 CQR：随 t 推进，新到达标签加入窗口
"""
from __future__ import annotations

import numpy as np


TARGET_COVERAGE = 0.90


def nonconformity_score(y_low, y_up, y_true):
    """s = max(y_low - y_true, y_true - y_up, 0)。"""
    return np.maximum(np.maximum(y_low - y_true, y_true - y_up), 0)


class FixedCQR:
    """固定 CQR：用 calib 集算 q_hat，区间 = [y_low - q_hat, y_up + q_hat]。"""

    def __init__(self, target=TARGET_COVERAGE):
        self.target = target
        self.q_hat = None

    def fit(self, calib_low, calib_up, calib_true):
        """calib 集算 q_hat（(1-α) 分位的非一致性分数，α=1-target）。"""
        s = nonconformity_score(calib_low, calib_up, calib_true)
        alpha = 1 - self.target
        self.q_hat = float(np.quantile(s, 1 - alpha))
        return self

    def calibrate(self, y_low, y_up):
        """返回校准区间 [y_low - q_hat, y_up + q_hat]。"""
        if self.q_hat is None:
            return y_low, y_up
        return y_low - self.q_hat, y_up + self.q_hat


class DelayedLabelSlidingCQR:
    """延迟标签感知滑动 CQR（§5.5）。

    时刻 t 的校准窗口 C_t = {τ: a_{τ+h} ≤ t 且 m_{τ+h}=1}。
    永久缺失标签不进窗口。合法标签不足时沿用最近 q_hat；无历史则回退（q_hat=0，用原始区间）。
    """

    def __init__(self, target=TARGET_COVERAGE, h=1, window_size=500):
        self.target = target
        self.h = h
        self.window_size = window_size  # 滑动窗口最大长度
        self.alpha = 1 - target
        self.scores = []  # 滑动窗口内的非一致性分数（按时序）
        self.last_q_hat = 0.0  # 沿用最近有效

    def update(self, tau, pred_low, pred_up, y_true_arrived, m_observed):
        """时刻推进：若 τ 时刻发出的 h 步预测的真值已到达且可观测，加入窗口。

        tau: 预测发出时刻
        pred_low/up: 该预测的原始区间
        y_true_arrived: τ+h 时刻的真值（若已到达）
        m_observed: τ+h 标签是否可观测（mask）
        返回更新后的 q_hat。
        """
        if y_true_arrived is not None and m_observed:
            s = nonconformity_score(pred_low, pred_up, y_true_arrived)
            self.scores.append(float(s))
            if len(self.scores) > self.window_size:
                self.scores = self.scores[-self.window_size:]
        if len(self.scores) > 0:
            self.last_q_hat = float(np.quantile(self.scores, 1 - self.alpha))
        return self.last_q_hat

    def calibrate_at(self, t, pred_low, pred_up):
        """时刻 t 的校准区间（用当前 last_q_hat）。"""
        return pred_low - self.last_q_hat, pred_up + self.last_q_hat
