"""时间模式提取。

- extract_global_intensity: S(t) = Σ Y_{od,t}，全局总强度。
- extract_od_relative_pattern: p_{od}(t) = Y_{od,t}/S(t)，相对 OD 结构。
- winsorize_intensity: 对 S(t) 做温和异常截断（消除测量噪声尖峰）。
"""
from __future__ import annotations

import numpy as np


def extract_global_intensity(matrix: np.ndarray) -> np.ndarray:
    """S(t) = matrix.sum(axis=(1,2))，长度 T，单位 Mbps。"""
    return matrix.sum(axis=(1, 2)).astype(np.float64)


def extract_od_relative_pattern(matrix: np.ndarray) -> np.ndarray:
    """p_{od}(t) = Y_{od,t} / S(t)，形状 (T,N,N)。S(t)=0 时刻置 0。"""
    S = matrix.sum(axis=(1, 2), keepdims=True)  # (T,1,1)
    safe = np.where(S > 0, S, 1.0)
    p = matrix / safe
    p[S.squeeze() == 0] = 0.0
    return p


def winsorize_intensity(S: np.ndarray, percentile: float = 99.9) -> tuple[np.ndarray, dict]:
    """对 S(t) 做温和异常截断。

    GÉANT/Abilene 存在测量噪声尖峰（如 GÉANT 有单点达 median×10000），
    会扭曲后续重采样与统计。截断到给定分位，保留真实突发（99 分位以下全保留）。

    Returns:
      (S_clipped, stats) 其中 stats 含原始/截断后的 min/max/median 与截断阈值、
      被截断点数。
    """
    thr = float(np.percentile(S, percentile))
    n_clipped = int((S > thr).sum())
    S_clipped = np.minimum(S, thr).astype(np.float64)
    stats = {
        "percentile": percentile,
        "threshold": thr,
        "n_clipped": n_clipped,
        "orig_min": float(S.min()),
        "orig_max": float(S.max()),
        "orig_median": float(np.median(S)),
        "clip_min": float(S_clipped.min()),
        "clip_max": float(S_clipped.max()),
        "clip_median": float(np.median(S_clipped)),
    }
    return S_clipped, stats
