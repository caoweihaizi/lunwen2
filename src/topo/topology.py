"""计划拓扑与星间链路可见性。

四邻居结构：2 轨内（同面 ±1）+ 2 跨轨（相邻面最近可见）。
"""
from __future__ import annotations

import numpy as np

from .orbit import RE, C_LIGHT


def _ecef_xyz(lon_deg, lat_deg, alt_km):
    """地心经纬度(度)+高程 → ECEF 笛卡尔坐标 km。"""
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    r = RE + alt_km
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def isl_visible(pos_i, pos_j, max_dist_km, alt_km):
    """两星间是否有视距链路（不被地球遮挡 + 距离阈值）。"""
    p_i = _ecef_xyz(*pos_i)
    p_j = _ecef_xyz(*pos_j)
    diff = p_j - p_i
    dist = np.linalg.norm(diff)
    if dist > max_dist_km:
        return False, dist
    # 判地球遮挡：连线最近点距地心 > Re
    # 参数化 P(t)=p_i + t*(p_j-p_i)，t∈[0,1]
    # 最近点 t* = -dot(p_i, diff)/|diff|^2
    t = -np.dot(p_i, diff) / np.dot(diff, diff)
    t = np.clip(t, 0.0, 1.0)
    closest = p_i + t * diff
    r_min = np.linalg.norm(closest)
    return r_min > RE, dist


def build_planned_topology(positions: np.ndarray, cfg) -> dict:
    """构造某时刻的四邻居计划拓扑。

    positions: (n_sat, 3) [lon,lat,alt]
    返回: adj (n,n) bool, distance_km (n,n) float, prop_delay_ms (n,n) float
    """
    n = len(positions)
    planes = int(cfg.constellation.walker.planes)
    spp = int(cfg.constellation.walker.sats_per_plane)
    max_dist = float(cfg.constellation.max_isl_distance_km)

    adj = np.zeros((n, n), dtype=bool)
    dist = np.zeros((n, n), dtype=np.float64)
    delay = np.zeros((n, n), dtype=np.float64)

    xyz = _ecef_xyz(positions[:, 0], positions[:, 1], positions[:, 2])

    def add_edge(i, j):
        d = np.linalg.norm(xyz[j] - xyz[i])
        if d > max_dist:
            return
        # 视距（LEO 间通常不被遮挡，仍判一下）
        diff = xyz[j] - xyz[i]
        t = -np.dot(xyz[i], diff) / max(np.dot(diff, diff), 1e-9)
        t = np.clip(t, 0.0, 1.0)
        r_min = np.linalg.norm(xyz[i] + t * diff)
        if r_min <= RE:
            return
        adj[i, j] = True
        adj[j, i] = True
        dist[i, j] = d
        dist[j, i] = d
        delay[i, j] = d / C_LIGHT * 1000.0  # km/(km/s)*1000 = ms
        delay[j, i] = delay[i, j]

    # 轨内邻居：同面 ±1
    for p in range(planes):
        for s in range(spp):
            i = p * spp + s
            nxt = p * spp + (s + 1) % spp
            prv = p * spp + (s - 1) % spp
            add_edge(i, nxt)
            add_edge(i, prv)

    # 跨轨邻居：每颗卫星只向"右面"(dp=+1)找最近可见邻居加一条边。
    # 由于 add_edge 对称，左面邻居由左面卫星向其右面(=本面)加边时获得，
    # 避免双向各加不同候选导致度数超 4。结果：每星 2 轨内 + 2 跨轨 = 4。
    for p in range(planes):
        for s in range(spp):
            i = p * spp + s
            p2 = (p + 1) % planes  # 右面
            cands = [p2 * spp + s2 for s2 in range(spp)]
            cand_d = [np.linalg.norm(xyz[c] - xyz[i]) for c in cands]
            order = np.argsort(cand_d)
            for k in order:
                c = cands[k]
                if adj[i, c]:
                    break
                before = adj[i, c]
                add_edge(i, c)
                if adj[i, c] and not before:
                    break

    # 后处理：强制每颗卫星度数 ≤ neighbors（默认 4）。
    # 跨轨几何可能使某星收到多余跨轨边，保留最近邻、删最远的跨轨边。
    max_deg = int(cfg.constellation.neighbors)
    for i in range(n):
        if adj[i].sum() <= max_deg:
            continue
        # 区分轨内（同面）与跨轨，只删跨轨中最远的
        p_i = i // spp
        cross = [j for j in range(n) if adj[i, j] and (j // spp) != p_i]
        cross.sort(key=lambda j: -dist[i, j])  # 从远到近删
        need_drop = int(adj[i].sum() - max_deg)
        for j in cross[:need_drop]:
            adj[i, j] = adj[j, i] = False
            dist[i, j] = dist[j, i] = 0.0
            delay[i, j] = delay[j, i] = 0.0

    return {
        "adj": adj,
        "distance_km": dist,
        "prop_delay_ms": delay,
    }
