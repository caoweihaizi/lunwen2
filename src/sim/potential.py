"""运行拓扑上的势函数（技术大纲 §6.3）。

故障已知后，在运行拓扑上重算 φ_op(i,d) = dist_{G_op}(i,d)。
正常时用计划拓扑势函数 φ_orb。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


def potential_hops_on_adj(adj: np.ndarray, dst: int) -> np.ndarray:
    """BFS 最短跳数（运行拓扑邻接）。dst 自身 0，不可达 -1。"""
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=np.int32)
    dist[dst] = 0
    from collections import deque
    q = deque([dst])
    while q:
        i = q.popleft()
        for j in np.where(adj[i])[0]:
            if dist[j] < 0:
                dist[j] = dist[i] + 1
                q.append(j)
    return dist


def all_pairs_hops(adj: np.ndarray) -> np.ndarray:
    """全对最短跳数矩阵 (n,n)，不可达 = 大数。"""
    n = adj.shape[0]
    D = csgraph.shortest_path(sp.csr_matrix(adj), directed=False, unweighted=True)
    D = np.where(np.isinf(D), 1e6, D)
    return D.astype(np.int32)
