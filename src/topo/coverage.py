"""卫星对地面像元的覆盖映射。

最大仰角选主服务星。几何判据：地心角距 ≤ 覆盖角。
"""
from __future__ import annotations

import numpy as np

from .orbit import RE


def _geocentric_angle(lon1, lat1, lon2, lat2):
    """两点地心角距（弧度）。"""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    cos_c = (np.sin(lat1) * np.sin(lat2) +
             np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))
    cos_c = np.clip(cos_c, -1.0, 1.0)
    return np.arccos(cos_c)


def coverage_half_angle(alt_km, min_elevation_deg):
    """覆盖半角（地心角）：球面几何，给定仰角阈值。"""
    el = np.deg2rad(min_elevation_deg)
    eta = np.arcsin(RE / (RE + alt_km))  # 地平角
    # 覆盖半角 theta = arccos(Re*cos(el)/(Re+alt)) - el
    theta = np.arccos(RE * np.cos(el) / (RE + alt_km)) - el
    return theta


def satellite_coverage(sat_pos, ground_lon, ground_lat, cfg):
    """单星对地面点集的覆盖布尔数组。

    sat_pos: (lon,lat,alt)
    ground_lon/lat: (K,) 地面点
    返回 (K,) bool
    """
    alt = sat_pos[2]
    min_el = float(cfg.constellation.min_elevation_deg)
    theta = coverage_half_angle(alt, min_el)
    c = _geocentric_angle(sat_pos[0], sat_pos[1], ground_lon, ground_lat)
    return c <= theta


def elevation_angle(sat_pos, ground_lon, ground_lat):
    """卫星对地面点的仰角（度）。用于选主服务星（最大仰角）。"""
    alt = sat_pos[2]
    c = _geocentric_angle(sat_pos[0], sat_pos[1], ground_lon, ground_lat)
    # 仰角 el = arctan( (cos(c)*(Re+alt) - Re) / (sin(c)*(Re+alt)) )
    num = np.cos(c) * (RE + alt) - RE
    den = np.sin(c) * (RE + alt)
    den_safe = np.where(den > 1e-9, den, 1e-9)
    el = np.arctan(num / den_safe)
    # c=0（正下方）仰角=90
    el = np.where(den <= 1e-9, np.pi / 2, el)
    return np.rad2deg(el)


def assign_primary_sat(positions, ground_lon, ground_lat, cfg):
    """给每个地面点分配主服务星（最大仰角）。

    positions: (n_sat, 3)
    返回 (K,) int 卫星 id，(K,) float 最大仰角
    """
    n_sat = len(positions)
    K = len(ground_lon)
    best_el = np.full(K, -np.inf, dtype=np.float64)
    best_sat = np.full(K, -1, dtype=np.int32)
    for s in range(n_sat):
        el = elevation_angle(positions[s], ground_lon, ground_lat)
        upd = el > best_el
        best_el = np.where(upd, el, best_el)
        best_sat = np.where(upd, s, best_sat)
    return best_sat, best_el
