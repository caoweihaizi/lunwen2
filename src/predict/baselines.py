"""预测基线：Historical Average / LSTM / GCN-GRU。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.common import resolve_paths
from .dataset import build_global_edge_set, preprocess_run, COL_T


def historical_average_predict(cfg, run, mechanism, edge_idx, split, normalize):
    """Historical Average：每条边用训练集同时段均值预测。"""
    p = resolve_paths(cfg)
    cache_path = p["data_processed"] / "predict" / "cache" / f"{run}_{mechanism}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        feat, truth = d["feat"], d["truth"]
    else:
        feat, truth = preprocess_run(cfg, run, mechanism, edge_idx, cache_path)

    import pickle
    times = pickle.load(open(p["data_interim"] / "topo" / "times.pkl", "rb"))["times"]
    train_s, train_e = split["train"]
    hours = np.array([t.hour for t in times])
    train_hours = hours[train_s:train_e]
    train_truth = truth[train_s:train_e]
    hour_mean = np.zeros((24, truth.shape[1]), dtype=np.float32)
    for h in range(24):
        mask = train_hours == h
        if mask.any():
            hour_mean[h] = train_truth[mask].mean(axis=0)
        else:
            hour_mean[h] = train_truth.mean(axis=0)

    test_s, test_e = split["test"]
    test_hours = hours[test_s:test_e]
    test_truth = truth[test_s:test_e]
    mean_c = normalize["mean"][0]
    std_c = normalize["std"][0]
    pred = hour_mean[test_hours]
    pred_norm = (pred - mean_c) / std_c
    truth_norm = (test_truth - mean_c) / std_c
    active = feat[test_s:test_e, :, 11] > 0
    return pred_norm, truth_norm, active


class LSTMBaseline(nn.Module):
    """单层 LSTM + 全连接头，无图、无缺失感知。"""

    def __init__(self, n_feat=12, hidden=64, n_h=2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True, num_layers=1)
        self.head = nn.Linear(hidden, n_h)  # 只预测中位数（点预测）

    def forward(self, x, future=None):
        out, h = self.lstm(x)
        feat = h[0][-1]  # (B, hidden)
        return self.head(feat)  # (B, n_h) 中位数预测


class GCNGRUBaseline(nn.Module):
    """简化 GCN-GRU：图卷积用固定邻接（无 attention），GRU 编码。

    简化：单边样本无图结构，GCN 退化为对边特征的线性变换（与本文模型同等简化）。
    """

    def __init__(self, n_feat=12, hidden=64, n_h=2):
        super().__init__()
        self.fc_in = nn.Linear(n_feat, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, num_layers=1)
        self.head = nn.Linear(hidden, n_h)

    def forward(self, x, future=None):
        h = self.fc_in(x)
        out, hh = self.gru(h)
        feat = hh[-1]
        return self.head(feat)


def train_baseline(model, train_ds, val_ds, log, epochs=20, batch_size=256, lr=1e-3, device="mps"):
    """训练点预测基线（MSE loss，预测中位数）。"""
    from torch.utils.data import DataLoader
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=batch_size)
    best = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        for x, f, y in tl:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vloss = 0.0; n = 0
        with torch.no_grad():
            for x, f, y in vl:
                x, y = x.to(device), y.to(device)
                vloss += loss_fn(model(x), y).item() * len(x); n += len(x)
        vloss /= max(n, 1)
        if vloss < best:
            best = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            log.info(f"  baseline epoch {ep+1}: val_mse {vloss:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    return model, best

