"""重采样到决策时隙 Δt。

只重采样一维 S(t) 序列（P3 只借 GÉANT 的时间模式，不借 OD 空间结构）。
线性插值，按时间戳数值化；不外推到原始范围外。
"""
from __future__ import annotations

from datetime import datetime

import numpy as np


def _to_epoch_minutes(times) -> np.ndarray:
    """datetime 列表 → 自首个时刻起的分钟数数组。"""
    base = times[0]
    return np.array([(t - base).total_seconds() / 60.0 for t in times], dtype=np.float64)


def resample_to_timeslot(S, times, target_minutes: int):
    """线性插值重采样 S(t) 到 target_minutes 粒度。

    Returns:
      (Sr: np.ndarray, target_times: list[datetime])
    """
    target_minutes = int(target_minutes)
    x = _to_epoch_minutes(times)
    base = times[0]
    last = times[-1]
    total_min = (last - base).total_seconds() / 60.0

    # 目标时刻（分钟数），从 0 到 total_min，步长 target_minutes
    target_x = np.arange(0.0, total_min + 1e-9, target_minutes, dtype=np.float64)
    Sr = np.interp(target_x, x, S)

    target_times = []
    for mx in target_x:
        target_times.append(
            datetime.fromtimestamp(base.timestamp() + mx * 60.0)
        )
    return Sr, target_times
