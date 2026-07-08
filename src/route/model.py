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
        # 边特征编码
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_edge_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # global + 聚合边特征 → logits
        self.head = nn.Sequential(
            nn.Linear(hidden + n_global, hidden), nn.ReLU(),
            nn.Linear(hidden, N_MAX_NEIGHBORS),
        )

    def forward(self, global_feat, edge_feat, mask):
        """
        global_feat: (B, n_global)
        edge_feat: (B, 4, n_edge_feat)
        mask: (B, 4) bool
        返回 logits (B, 4)，已 mask。
        """
        # 边特征编码后取 mean 聚合
        B = edge_feat.shape[0]
        e = self.edge_mlp(edge_feat)  # (B, 4, hidden)
        e_agg = e.mean(dim=1)  # (B, hidden)
        combined = torch.cat([e_agg, global_feat], dim=-1)  # (B, hidden+n_global)
        logits = self.head(combined)  # (B, 4)
        # mask：无效位置 -inf
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class Critic(nn.Module):
    """centralized critic：全局链路状态 → V。"""

    def __init__(self, n_global_state, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_global_state, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state):
        return self.net(global_state).squeeze(-1)
