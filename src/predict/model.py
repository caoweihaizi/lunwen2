"""Mask-aware GAT-GRU-D 预测模型（技术大纲 §5.1-5.4）。

输入：历史窗口 W=30 时隙的边特征 (B, W, N_FEAT) + 未来计划 (B, H_max, 3)
输出：每条边未来 h=1,3 的 0.05/0.50/0.95 分位数 (B, n_H, 3)

架构（轻量版，P7 决策）：
  - GRU-D 缺失恢复（§5.2）
  - 轻量 GRU 编码时序（§5.4，简化：不显式建节点 GAT，因每样本单边无图结构；
    GAT 在"流级"训练时改为边级特征聚合——见说明）
  - 单调分位数头（§5.4）

说明：技术大纲 §5.3 的节点 GAT 需要整图（同节点多边消息传递）。本实现按"单边样本"
训练，GAT 退化为边特征 MLP（两端节点特征缺失）。完整 GAT 在 P10 路由环境里实现
（那时需要候选邻居集合）。P7 预测器专注边级时序 + 缺失恢复，是合理简化。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUDRecovery(nn.Module):
    """GRU-D 式缺失恢复（§5.2）。

    γ_t = exp(-max(0, W_γ·Δ_t + b_γ))
    x̃_t = m_t·x_t + (1-m_t)[γ_t·x_last + (1-γ_t)·x_mean]
    mask/age 作为显式特征保留。
    """

    def __init__(self, n_feat):
        super().__init__()
        # 对每个缺失字段学衰减率：输入 3 字段 age → 输出 3 字段 gamma
        self.W_gamma = nn.Linear(3, 3, bias=True)
        self.register_buffer("x_mean", torch.zeros(3))

    def forward(self, x, mask, age):
        """
        x: (B, W, N_FEAT)（缺失处已置0）
        mask: (B, W, 3)（1=观测）
        age: (B, W, 3) 信息年龄
        返回恢复后的 (B, W, N_FEAT)，前3列被恢复
        """
        # γ = exp(-max(0, W·age + b))，每字段独立
        gamma = torch.exp(-torch.clamp(self.W_gamma(age), min=0))  # (B,W,3)
        x_obs = x[..., :3]  # (B,W,3)
        x_rec = mask * x_obs + (1 - mask) * ((1 - gamma) * self.x_mean)
        x_out = x.clone()
        x_out[..., :3] = x_rec
        return x_out


class PredictModel(nn.Module):
    """Mask-aware GRU-D + 轻量时序解码 + 单调分位数头。

    输入: x (B,W,N_FEAT), future (B,H_max,3)
    输出: quantiles (B, n_H, 3) [q05, q50, q95] 每个预测时域
    """

    def __init__(self, n_feat=12, hidden=64, n_h=2, h_max=3, n_gru_layers=1):
        super().__init__()
        self.grud = GRUDRecovery(n_feat)
        self.gru = nn.GRU(input_size=n_feat, hidden_size=hidden,
                          batch_first=True, num_layers=n_gru_layers,
                          dropout=0.1 if n_gru_layers > 1 else 0)
        # 未来计划编码
        self.future_mlp = nn.Linear(3, hidden)
        # 解码
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # 单调分位数头：每个预测时域输出 3 个 raw 值 z0,z1,z2
        self.quantile_head = nn.Linear(hidden, n_h * 3)
        self.n_h = n_h
        self.h_max = h_max

    def forward(self, x, future):
        """
        x: (B, W, N_FEAT)
        future: (B, H_max, 3)
        返回 quantiles: (B, n_h, 3)
        """
        mask = x[..., 3:6]   # mask_carried/util/queue
        age = x[..., 6:9]    # age
        x_rec = self.grud(x, mask, age)
        # GRU 编码
        out, h = self.gru(x_rec)  # out: (B,W,hidden), h: (n_layers,B,hidden)
        feat = h[-1]  # 取最后一层 (B, hidden)
        # 未来计划：取最后一个时域（h=H_max）
        fut_feat = self.future_mlp(future[:, -1, :])  # (B, hidden)
        combined = torch.cat([feat, fut_feat], dim=-1)  # (B, 2*hidden)
        dec = self.decoder(combined)  # (B, hidden)
        raw = self.quantile_head(dec)  # (B, n_h*3)
        raw = raw.view(-1, self.n_h, 3)  # (B, n_h, 3)
        # 单调头：q05 可负（标准化后标签有负值），q50/q95 用 softplus 增量保证单调
        q05 = raw[..., 0]  # 可负
        q50 = q05 + torch.nn.functional.softplus(raw[..., 1])
        q95 = q50 + torch.nn.functional.softplus(raw[..., 2])
        quantiles = torch.stack([q05, q50, q95], dim=-1)  # (B, n_h, 3)
        return quantiles


def pinball_loss(pred, target, quantiles=(0.05, 0.50, 0.95)):
    """Pinball Loss（§5.4）。

    pred: (B, n_h, 3) 预测分位数
    target: (B, n_h) 真值
    """
    losses = []
    for k, q in enumerate(quantiles):
        err = target - pred[..., k]  # (B, n_h)
        loss = torch.max(q * err, (q - 1) * err)
        losses.append(loss.mean())
    return sum(losses) / len(losses)
