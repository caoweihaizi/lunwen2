"""预测评价：MAE/RMSE/MAPE/峰值MAE（原始量纲）。"""
from __future__ import annotations

import numpy as np


def evaluate_point(pred_norm, truth_norm, active, mean_c, std_c):
    """点预测评价（中位数）。pred/truth 在标准化空间，active 是 bool 掩码。

    返回 MAE/RMSE/MAPE/peak_MAE（原始 Mbps 量纲）。
    """
    p = pred_norm * std_c + mean_c
    t = truth_norm * std_c + mean_c
    if active is not None:
        p = p[active]; t = t[active]
    err = np.abs(p - t)
    mae = float(err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    # MAPE：避免除零
    mape = float(np.mean(err / np.maximum(np.abs(t), 1.0)) * 100)
    # 峰值 MAE：真值 top 10% 的误差
    thr = np.percentile(np.abs(t), 90)
    peak_mask = np.abs(t) >= thr
    peak_mae = float(err[peak_mask].mean()) if peak_mask.any() else 0.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "peak_MAE": peak_mae,
            "n_samples": int(len(p))}
