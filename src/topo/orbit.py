"""Walker 星座轨道生成器（简化 J2 圆球解析模型）。

物理简化：
- 圆轨道，J2 仅影响升交点赤经漂移（RAAN drift）；
- 卫星位置用球面三角解析解，真近点角线性化；
- 球面地球（Re=6371 km）。

Walker Delta 6/66/F：6 面 11 星，面间相位偏移 F。
"""
from __future__ import annotations

import numpy as np

# 物理常数
MU = 3.986e5  # km^3/s^2
RE = 6371.0   # km
J2 = 1.0826e-3
C_LIGHT = 299792.458  # km/s


def _deg2rad(d):
    return np.deg2rad(d)


class WalkerConstellation:
    """Walker Delta 星座。

    cfg.constellation.walker: {planes, sats_per_plane, inclination_deg, altitude_km, f}
    """

    def __init__(self, cfg):
        w = cfg.constellation.walker
        self.planes = int(w.planes)
        self.sats_per_plane = int(w.sats_per_plane)
        self.n_sat = self.planes * self.sats_per_plane
        self.inc = _deg2rad(float(w.inclination_deg))
        self.alt = float(w.altitude_km)  # km
        self.f = int(getattr(w, "f", 2))
        self.a = RE + self.alt  # 轨道半长轴 km
        self.n = np.sqrt(MU / self.a ** 3)  # 平均运动 rad/s
        self.period = 2 * np.pi / self.n  # s
        # J2 RAAN 漂移 rad/s
        self.raan_dot = -1.5 * J2 * (RE / self.a) ** 2 * self.n / np.cos(self.inc)

        # 每颗卫星的初始 RAAN、初始纬度幅角 u0
        self.raan0 = np.zeros(self.n_sat)
        self.u0 = np.zeros(self.n_sat)
        plane_phase = 2 * np.pi / self.planes
        sat_phase = 2 * np.pi / self.sats_per_plane
        f_phase = 2 * np.pi * self.f / self.n_sat  # 面间相位偏移
        for p in range(self.planes):
            for s in range(self.sats_per_plane):
                sid = p * self.sats_per_plane + s
                self.raan0[sid] = p * plane_phase
                self.u0[sid] = s * sat_phase + p * f_phase

    def position(self, sat_id: int, t_seconds: float):
        """返回 (lon, lat) 地心经纬度（度），alt km。"""
        raan = self.raan0[sat_id] + self.raan_dot * t_seconds
        u = self.u0[sat_id] + self.n * t_seconds  # 纬度幅角
        # 轨道平面内：卫星位置（惯性系球坐标）
        # lat = arcsin(sin(i) * sin(u))
        lat = np.arcsin(np.sin(self.inc) * np.sin(u))
        # 经度（含地球自转）：alpha = raan + arctan2(cos(i)*sin(u), cos(u))
        # 地球自转：本仿真用惯性经度（不考虑地球自转，因为 P3 用世界时昼夜调制而非卫星-地面相对运动）
        # 但覆盖映射需要 ECEF 经度，所以减去地球自转角
        omega_earth = 7.2921e-5  # rad/s
        arg = np.arctan2(np.cos(self.inc) * np.sin(u), np.cos(u))
        lon = raan + arg - omega_earth * t_seconds
        # 归一化到 [-180, 180]
        lon = ((lon + np.pi) % (2 * np.pi)) - np.pi
        return np.rad2deg(lon), np.rad2deg(lat), self.alt

    def all_positions(self, t_seconds: float) -> np.ndarray:
        """返回 (n_sat, 3) [lon, lat, alt]。向量化。"""
        t = float(t_seconds)
        raan = self.raan0 + self.raan_dot * t
        u = self.u0 + self.n * t
        sin_i = np.sin(self.inc)
        cos_i = np.cos(self.inc)
        lat = np.arcsin(sin_i * np.sin(u))
        arg = np.arctan2(cos_i * np.sin(u), np.cos(u))
        omega_earth = 7.2921e-5
        lon = raan + arg - omega_earth * t
        lon = ((lon + np.pi) % (2 * np.pi)) - np.pi
        out = np.zeros((self.n_sat, 3), dtype=np.float64)
        out[:, 0] = np.rad2deg(lon)
        out[:, 1] = np.rad2deg(lat)
        out[:, 2] = self.alt
        return out
