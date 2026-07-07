"""整段仿真时间的拓扑时间序列。

与 P1 resampled_times 对齐。稀疏存储：每时隙 edge_list + edge_dist + edge_delay。
"""
from __future__ import annotations

import numpy as np

from .orbit import WalkerConstellation
from .topology import build_planned_topology


def _to_epoch_seconds(times):
    """datetime 列表 → 自首时刻起的秒数。"""
    base = times[0]
    return np.array([(t - base).total_seconds() for t in times], dtype=np.float64)


def generate_topology_series(cfg, times):
    """生成与 times 对齐的拓扑时间序列。

    times: list[datetime]（来自 P1 resampled_times）
    返回:
      {
        "positions": (T, n_sat, 3) float32,
        "edge_lists": list of (E_t, 2) int32,
        "edge_dists": list of (E_t,) float32,   # km
        "edge_delays": list of (E_t,) float32,  # ms
        "sat_ids": list[int],
      }
    """
    wc = WalkerConstellation(cfg)
    n_sat = wc.n_sat
    T = len(times)
    epoch = _to_epoch_seconds(times)

    positions = np.zeros((T, n_sat, 3), dtype=np.float32)
    edge_lists, edge_dists, edge_delays = [], [], []

    for t in range(T):
        pos = wc.all_positions(float(epoch[t]))
        positions[t] = pos
        topo = build_planned_topology(pos, cfg)
        adj = topo["adj"]
        # 提取上三角边（无向，避免重复）
        ii, jj = np.where(np.triu(adj, k=1))
        edges = np.stack([ii, jj], axis=1).astype(np.int32)
        edge_lists.append(edges)
        edge_dists.append(topo["distance_km"][ii, jj].astype(np.float32))
        edge_delays.append(topo["prop_delay_ms"][ii, jj].astype(np.float32))

    return {
        "positions": positions,
        "edge_lists": edge_lists,
        "edge_dists": edge_dists,
        "edge_delays": edge_delays,
        "sat_ids": list(range(n_sat)),
    }
