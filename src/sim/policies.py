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
    stateful: 需当前利用率 ρ、队列 q。
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, eta=0.1, T=1.0):
        self.name = "queue_aware"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.T = T

    def prepare(self, net, delays_unused):
        self.hops = all_pairs_hops(net.adj)
        self.phi = self.hops.astype(np.float64)

    def decide(self, net, commodity_active, t, rng):
        out = {}
        adj = net.adj
        # 当前利用率/队列（归一化）
        cap = net.links[0].capacity if net.links else 1.0
        for (i, d) in commodity_active:
            if i == d:
                out[(i, d)] = {}
                continue
            phi_d = self.hops[:, d]
            target = phi_d[i] - 1
            cands = [j for j in np.where(adj[i])[0] if phi_d[j] == target]
            if not cands:
                out[(i, d)] = {}
                continue
            # 算每个候选的代价
            costs = []
            for j in cands:
                lk = net.get_link(i, j)
                rho = lk.queue / max(lk.capacity, 1e-9)  # 利用率近似
                q = lk.queue / max(lk.buffer, 1e-9)
                delta = net.link_delay[net.edge_idx[(i, j)]]
                c = (self.alpha * phi_d[j] + self.beta * rho
                     + self.gamma * q + self.eta * delta)
                costs.append(c)
            costs = np.array(costs)
            # 软max 采样
            p = np.exp(-costs / self.T)
            p = p / p.sum()
            # 采样一个下一跳（stochastic）——但为分流记录，按概率作为 split_ratio
            # 技术大纲：按温度采样。这里输出概率作为 split_ratio（期望分流）
            out[(i, d)] = {int(j): float(p[k]) for k, j in enumerate(cands)}
        return out
