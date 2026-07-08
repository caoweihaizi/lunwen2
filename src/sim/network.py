"""网络状态：从稀疏边表重建邻接 + 链路属性，持有有向链路队列。

故障链路从邻接移除，不承载数据。
"""
from __future__ import annotations

import numpy as np

from .queue import QueueState, capacity_per_slot


class NetworkState:
    """某时隙的网络状态。

    每条有向链路一个 QueueState。故障链路不存在。
    """

    def __init__(self, edges, dists, delays, n_sat, cfg, dt_min, failed_edges=None):
        """
        edges: (E,2) int 计划拓扑无向边表
        dists/delays: (E,) 距离km/时延ms
        failed_edges: set of frozenset({i,j}) 故障边
        """
        self.n_sat = n_sat
        self.dt_min = dt_min
        self.cap_gbps = float(cfg.constellation.link_capacity_gbps)
        cap_mbs = capacity_per_slot(self.cap_gbps, dt_min)
        # B = 1 时隙容量
        buffer_mbs = cap_mbs * 1.0
        max_wait = 2

        failed_edges = failed_edges or set()
        # 建有向链路：每条无向边 → 2 有向，跳过故障
        self.adj = np.zeros((n_sat, n_sat), dtype=bool)
        self.edge_idx = {}  # (i,j) -> 索引
        self.links = []     # list of QueueState
        self.link_dist = []
        self.link_delay = []
        self.link_endpoints = []
        self.failed = set()

        for k in range(len(edges)):
            i, j = int(edges[k, 0]), int(edges[k, 1])
            if frozenset({i, j}) in failed_edges:
                self.failed.add(frozenset({i, j}))
                continue
            for (a, b) in [(i, j), (j, i)]:
                idx = len(self.links)
                self.links.append(QueueState(cap_mbs, buffer_mbs, max_wait))
                self.link_dist.append(float(dists[k]))
                self.link_delay.append(float(delays[k]))
                self.link_endpoints.append((a, b))
                self.edge_idx[(a, b)] = idx
                self.adj[a, b] = True

        self.n_links = len(self.links)

    def neighbors(self, i):
        return np.where(self.adj[i])[0]

    def get_link(self, i, j):
        return self.links[self.edge_idx[(i, j)]]

    def step_links(self, arrivals):
        """arrivals: dict {(i,j): mb}。返回 dict {(i,j): (served, dropped, queue, offered)}。"""
        results = {}
        for (i, j), arr in arrivals.items():
            if (i, j) not in self.edge_idx:
                continue
            lk = self.links[self.edge_idx[(i, j)]]
            served, dropped, q, offered = lk.step(arr)
            results[(i, j)] = (served, dropped, q, offered)
        return results
