"""总量标定 k0（技术大纲 §4.1.4）。

目标：调 k0 使链路利用率中位≈0.5、P95≈0.8。
方法：用最短路代理分流（Dijkstra），二分 k0。
P3 给初始 k0，P4 完整仿真后可 refine。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


def _rebuild_adj_delay(edge_list, edge_delay, n_sat):
    """稀疏边表 → 邻接 + 时延稀疏矩阵。"""
    if len(edge_list) == 0:
        return sp.csr_matrix((n_sat, n_sat))
    src = edge_list[:, 0]
    dst = edge_list[:, 1]
    d = edge_delay
    # 无向：双向都加
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    vals = np.concatenate([d, d])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_sat, n_sat))


def _shortest_path_flow(commodity_arr, dist_mat, n_sat):
    """最短路代理分流：每条 commodity 沿最短路径分配，累计各链路负载。

    commodity_arr: (M,4) [src,dst,vol,ftype]
    dist_mat: (n_sat,n_sat) 最短路径距离（用传播时延作权重）
    返回每条边的负载 dict {(i,j): load}，以及前驱用于路径还原。
    简化：用最短跳数路径，流量均分到所有最短路（这里简化为单条最短路）。
    """
    # 用 csgraph 的前驱矩阵还原路径
    # 为效率，先对所有目的算最短路 predecessor
    # 但 commodity 目的多样，逐条处理
    edge_load = {}
    for row in commodity_arr:
        s, d, v = int(row[0]), int(row[1]), float(row[2])
        if s == d or v <= 0:
            continue
        # 用 dist_mat 找最短路径（Dijkstra 前驱）
        # 简化：用最短跳数（BFS 邻接），流量沿一条最短路
        # 这里用 dist_mat 的 precomputed predecessor
        pass
    return edge_load


def estimate_k0(commodity_ts, topo_edge_lists, topo_edge_delays, times, cfg,
                sample_t=None):
    """估算 k0 使链路利用率中位≈0.5、P95≈0.8。

    用最短跳数路径分流（代理 P4 路由），采样若干时隙估计利用率分布。
    二分 k0。

    commodity_ts: 假设 k0=1 时的 commodity（baseline 场景）
    返回 (k0, util_stats)
    """
    n_sat = int(cfg.constellation.n_sat)
    cap_gbps = float(cfg.constellation.link_capacity_gbps)
    # 流量单位：S(t) 是 Mbps，commodity 也是 Mbps（k0=1 时）。
    # 链路容量 Gbps → Mbps
    cap_mbps = cap_gbps * 1000.0

    # 采样时隙（避免全跑）
    T = len(times)
    if sample_t is None:
        sample_t = np.linspace(0, T - 1, min(200, T)).astype(int)

    # 预计算采样时隙的最短跳数 predecessor（BFS，基于邻接）
    # 为每对 (src,dst) 算最短路径，需要前驱
    # 用 scipy dijkstra 一次算全源最短路 predecessor
    # 但邻接每时隙变，采样若干时隙各算一次
    def _util_at_k0(k0):
        utils = []
        for ti in sample_t:
            edges = topo_edge_lists[ti]
            if len(edges) == 0:
                continue
            # 邻接矩阵（无权，最短跳数）
            adj = sp.csr_matrix(
                (np.ones(len(edges) * 2),
                 (np.concatenate([edges[:, 0], edges[:, 1]]),
                  np.concatenate([edges[:, 1], edges[:, 0]]))),
                shape=(n_sat, n_sat),
            )
            # 全源最短路跳数 + 前驱
            D, Pr = csgraph.shortest_path(adj, directed=False, return_predecessors=True,
                                          unweighted=True)
            # 累计边负载
            edge_load = {}
            comm = commodity_ts[ti]
            for row in comm:
                s, d, v = int(row[0]), int(row[1]), float(row[2]) * k0
                if s == d or v <= 0 or not np.isfinite(D[s, d]):
                    continue
                # 沿前驱还原路径
                path = [d]
                cur = d
                while cur != s:
                    p = Pr[s, cur]
                    if p < 0:
                        break
                    path.append(p)
                    cur = p
                path.reverse()
                for a, b in zip(path[:-1], path[1:]):
                    key = (min(a, b), max(a, b))
                    edge_load[key] = edge_load.get(key, 0.0) + v
            if not edge_load:
                continue
            loads = np.array(list(edge_load.values()))
            utils.append(loads / cap_mbps)
        if not utils:
            return np.array([0.0]), np.array([])
        all_u = np.concatenate(utils)
        return all_u, all_u

    # 二分 k0：目标 P95≈0.8
    lo, hi = 1e-6, 10.0
    best_k0 = None
    for _ in range(20):
        mid = (lo + hi) / 2
        u, _ = _util_at_k0(mid)
        p95 = np.percentile(u, 95) if len(u) else 0
        if p95 < 0.8:
            lo = mid
        else:
            hi = mid
        best_k0 = mid
    # 最终统计
    u_final, _ = _util_at_k0(best_k0)
    util_stats = {
        "k0": float(best_k0),
        "util_median": float(np.median(u_final)) if len(u_final) else 0.0,
        "util_p95": float(np.percentile(u_final, 95)) if len(u_final) else 0.0,
        "util_mean": float(np.mean(u_final)) if len(u_final) else 0.0,
        "util_max": float(np.max(u_final)) if len(u_final) else 0.0,
        "n_samples": int(len(u_final)),
        "note": "P3 最短路代理标定，P4 完整仿真后可 refine",
    }
    return best_k0, util_stats
