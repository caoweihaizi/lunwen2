"""三类离线行为策略（技术大纲 §4.2 步骤 8）。

- Dijkstra: 最短传播时延，每节点下一跳=最短路上后继
- ECMP: 等价多路径，分叉点均分
- Queue-aware stochastic: 局部代价 + 软max采样（stateful，需当前队列/利用率）

每个策略对每个活跃 (i,d) 决定下一跳分配：
  返回 dict {(i,d): {j: split_ratio}}，split_ratio 和=1。
  无合法邻居则 {}（等待）。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

from .potential import all_pairs_hops


def _valid_next_hops(adj, phi_d, i):
    """合法下一跳：邻居 且 势值严格下降。"""
    nbrs = np.where(adj[i])[0]
    return [j for j in nbrs if phi_d[j] < phi_d[i]]


class DijkstraPolicy:
    """最短传播时延。每目的 d 全源最短路，下一跳=后继。"""

    def __init__(self):
        self.name = "dijkstra"

    def prepare(self, net: "NetworkState", delays_unused):
        """预计算每目的的势函数(最短时延)与前驱。"""
        # 用传播时延作权重的全源最短路
        n = net.n_sat
        rows, cols, vals = [], [], []
        for (a, b), idx in net.edge_idx.items():
            rows.append(a); cols.append(b); vals.append(net.link_delay[idx])
        G = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        self.D, self.Pr = csgraph.shortest_path(G, directed=False, return_predecessors=True)
        self.phi = self.D.copy()
        # 跳数势函数（供 e2e_delay 复用，避免重算）
        self.hops = all_pairs_hops(net.adj)

    def decide(self, net, commodity_active, t):
        """commodity_active: list of (i,d)。返回 {(i,d):{j:ratio}}。"""
        out = {}
        for (i, d) in commodity_active:
            if i == d or not np.isfinite(self.D[i, d]):
                out[(i, d)] = {}
                continue
            # 后继：Pr[d, i] 是 i 在到 d 最短路上的前驱，反过来 i 的下一跳
            nxt = self.Pr[d, i]
            if nxt < 0 or nxt == i:
                out[(i, d)] = {}
                continue
            # 校验势值下降
            if self.phi[nxt, d] < self.phi[i, d]:
                out[(i, d)] = {int(nxt): 1.0}
            else:
                out[(i, d)] = {}
        return out


class ECMPPolicy:
    """等价多路径。每目的 d 所有等价最短跳数路径，分叉均分。"""

    def __init__(self):
        self.name = "ecmp"

    def prepare(self, net, delays_unused):
        n = net.n_sat
        self.hops = all_pairs_hops(net.adj)  # (n,n) 最短跳数
        # 势函数 = 最短跳数
        self.phi = self.hops.astype(np.float64)

    def decide(self, net, commodity_active, t):
        out = {}
        adj = net.adj
        for (i, d) in commodity_active:
            if i == d:
                out[(i, d)] = {}
                continue
            phi_d = self.hops[:, d]
            # 等价下一跳：邻居 且 phi[j,d]==phi[i,d]-1
            target = phi_d[i] - 1
            cands = [j for j in np.where(adj[i])[0] if phi_d[j] == target]
            if not cands:
                out[(i, d)] = {}
                continue
            ratio = 1.0 / len(cands)
            out[(i, d)] = {int(j): ratio for j in cands}
        return out


class QueueAwareStochasticPolicy:
    """队列感知随机路由（技术大纲 §4.2 公式）。

    c_{ij,d,t} = α·φ(j,d) + β·ρ_{ij,t} + γ·q_{ij,t} + η·δ_{ij,t}
    P(j|i,d,t) = softmax(-c/T) over valid candidates.

    候选范围：势值下降（最优）或相等（带惩罚）的邻居。66 星稀疏拓扑下
    势值下降邻居常唯一，扩大到相等邻居让队列状态能把流量推离拥塞链路。
    注：势值严格下降是 P10 action masking 的约束（§6.3），离线策略不受此限。

    代价归一化：φ 归一到 [0,1]（除以 max hops），ρ/q 已在 [0,1]，δ 归一到 [0,1]。
    stateful: 需当前利用率 ρ、队列 q。
    """

    def __init__(self, alpha=1.0, beta=2.0, gamma=1.0, eta=0.3, T=0.3,
                 equal_potential_penalty=0.5):
        self.name = "queue_aware"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.T = T
        self.eq_pen = equal_potential_penalty

    def prepare(self, net, delays_unused):
        self.hops = all_pairs_hops(net.adj)
        self.phi = self.hops.astype(np.float64)
        self.max_hops = max(float(self.hops.max()), 1.0)
        self.max_delay = max(float(max(net.link_delay) if net.link_delay else 1.0), 1.0)

    def decide(self, net, commodity_active, t, rng):
        out = {}
        adj = net.adj
        for (i, d) in commodity_active:
            if i == d:
                out[(i, d)] = {}
                continue
            phi_d = self.hops[:, d]
            cur = phi_d[i]
            # 候选：势值下降（target=cur-1）或相等（cur）的邻居
            cands = []
            for j in np.where(adj[i])[0]:
                if phi_d[j] == cur - 1:
                    cands.append((j, 0.0))  # 下降，无惩罚
                elif phi_d[j] == cur:
                    cands.append((j, self.eq_pen))  # 相等，带惩罚
            if not cands:
                out[(i, d)] = {}
                continue
            # 算每个候选的归一化代价
            costs = []
            for j, pen in cands:
                lk = net.get_link(i, j)
                rho = lk.queue / max(lk.capacity, 1e-9)  # [0,1]
                q = lk.queue / max(lk.buffer, 1e-9)      # [0,1]
                delta = net.link_delay[net.edge_idx[(i, j)]] / self.max_delay  # [0,1]
                phi_norm = phi_d[j] / self.max_hops      # [0,1]
                c = (self.alpha * phi_norm + self.beta * rho
                     + self.gamma * q + self.eta * delta + pen)
                costs.append(c)
            costs = np.array(costs)
            p = np.exp(-costs / self.T)
            p = p / p.sum()
            out[(i, d)] = {int(j): float(p[k]) for k, (j, _) in enumerate(cands)}
        return out
