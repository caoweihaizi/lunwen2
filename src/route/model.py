"""MAPPO Actor-Critic（技术大纲 §6.1/6.6）。

Actor: (global, edge_features[4,F], mask) → logits[4]，masked softmax
Critic: 全局状态 → V(s)
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_EDGE_FEAT = 13
N_MAX_NEIGHBORS = 4


class Actor(nn.Module):
    def __init__(self, n_global=3, n_edge_feat=N_EDGE_FEAT, hidden=64):
        super().__init__()
        # 边特征编码（输入归一化：对每维特征做 LayerNorm 稳定量级）
        self.edge_norm = nn.LayerNorm(n_edge_feat)
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_edge_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # global + 聚合边特征 → logits
        self.head = nn.Sequential(
            nn.Linear(hidden + n_global, hidden), nn.ReLU(),
            nn.Linear(hidden, N_MAX_NEIGHBORS),
        )
        self.logit_scale = 1.0  # 限制 logit 量级，避免 softmax 退化

    def forward(self, global_feat, edge_feat, mask):
        """
        global_feat: (B, n_global)
        edge_feat: (B, 4, n_edge_feat)
        mask: (B, 4) bool
        返回 logits (B, 4)，已 mask。
        """
        # 输入归一化
        edge_feat = self.edge_norm(edge_feat)
        e = self.edge_mlp(edge_feat)  # (B, 4, hidden)
        e_agg = e.mean(dim=1)  # (B, hidden)
        combined = torch.cat([e_agg, global_feat], dim=-1)  # (B, hidden+n_global)
        logits = self.head(combined)  # (B, 4)
        logits = torch.tanh(logits) * self.logit_scale  # 限制量级，避免 softmax 退化
        # 防 nan（LayerNorm 在退化 batch 下可能产生 nan）
        logits = torch.nan_to_num(logits, nan=0.0)
        # mask：无效位置 -inf
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class Critic(nn.Module):
    """centralized critic：全局状态 + (i,d) 编码 → 该 (i,d) 的 V。

    标准多智能体 MAPPO：每个 agent (i,d) 独立估值，adv 按每个 (i,d) 算。
    """

    def __init__(self, n_global_state, n_sat=264, hidden=128, n_global=3):
        super().__init__()
        # 全局状态编码
        self.state_net = nn.Sequential(
            nn.Linear(n_global_state, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # (i,d) 编码：current_sat + dst_sat
        self.sat_embed = nn.Embedding(n_sat, 16)
        # 融合：全局状态 + (i,d) embedding + commodity global 特征
        self.head = nn.Sequential(
            nn.Linear(hidden + 16 * 2 + n_global, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state, cur_sat, dst_sat, global_feat):
        """
        global_state: (B, n_global_state) 全局链路状态
        cur_sat: (B,) 当前卫星 id
        dst_sat: (B,) 目的卫星 id
        global_feat: (B, n_global) commodity 全局特征
        返回 (B,) 每个 (i,d) 的 V
        """
        s = self.state_net(global_state)  # (B, hidden)
        cur_emb = self.sat_embed(cur_sat)  # (B, 16)
        dst_emb = self.sat_embed(dst_sat)  # (B, 16)
        combined = torch.cat([s, cur_emb, dst_emb, global_feat], dim=-1)
        return self.head(combined).squeeze(-1)
