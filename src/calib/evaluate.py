"""校准评价：PICP/MPIW/Coverage Gap/Winkler Score（§7.4）。"""
from __future__ import annotations

import numpy as np


def picp(y_low, y_up, y_true):
    """预测区间覆盖率。"""
    return float(np.mean((y_true >= y_low) & (y_true <= y_up)))


def mpiw(y_low, y_up):
    """平均预测区间宽度。"""
    return float(np.mean(y_up - y_low))


def coverage_gap(picp_val, target=0.90):
    """实际覆盖率与目标差。"""
    return float(picp_val - target)


def winkler_score(y_low, y_up, y_true, target=0.90):
    """Winkler Score：覆盖与宽度同时衡量（越小越好）。

    W = (y_up - y_low) + (2/α) * (y_low - y_true) if y_true < y_low
                         + (2/α) * (y_true - y_up)  if y_true > y_up
    α = 1 - target
    """
    alpha = 1 - target
    width = y_up - y_low
    penalty_low = np.where(y_true < y_low, (2 / alpha) * (y_low - y_true), 0)
    penalty_up = np.where(y_true > y_up, (2 / alpha) * (y_true - y_up), 0)
    return float(np.mean(width + penalty_low + penalty_up))


def evaluate_intervals(y_low, y_up, y_true, target=0.90):
    """全套指标。"""
    p = picp(y_low, y_up, y_true)
    return {
        "PICP": p,
        "MPIW": mpiw(y_low, y_up),
        "Coverage_Gap": coverage_gap(p, target),
        "Winkler": winkler_score(y_low, y_up, y_true, target),
        "n": int(len(y_true)),
    }
