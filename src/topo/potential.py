"""目的势函数预计算。

- potential_shortest_hops: BFS 最短跳数（计划拓扑）。
- potential_propagation_delay: Dijkstra 最小传播时延。
"""
from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse.csgraph as csgraph


def potential_shortest_hops(adj: np.ndarray, dst: int) -> np.ndarray:
    """BFS：每颗卫星到 dst 的最短跳数。dst 自身为 0，不可达为 -1。"""
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=np.int32)
    dist[dst] = 0
    q = deque([dst])
    while q:
        i = q.popleft()
        for j in np.where(adj[i])[0]:
            if dist[j] < 0:
                dist[j] = dist[i] + 1
                q.append(j)
    return dist


def potential_propagation_delay(prop_delays: np.ndarray, dst: int) -> np.ndarray:
    """Dijkstra：每颗卫星到 dst 的最小传播时延(ms)。用 scipy。"""
    n = prop_delays.shape[0]
    # 构稀疏图（仅边权>0）
    rows, cols, vals = [], [], []
    nz = np.argwhere(prop_delays > 0)
    for i, j in nz:
        rows.append(i); cols.append(j); vals.append(prop_delays[i, j])
    import scipy.sparse as sp
    G = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # csgraph.dijkstra 返回 (n,) 到 dst 的最短路
    d = csgraph.dijkstra(G, indices=dst, directed=False)
    d = np.where(np.isinf(d), -1.0, d)
    return d
