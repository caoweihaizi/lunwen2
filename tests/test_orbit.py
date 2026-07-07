"""Walker 星座轨道单元测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config
from src.topo.orbit import WalkerConstellation


def _cfg():
    return load_config(cli=False)


def test_n_sat_and_period():
    wc = WalkerConstellation(_cfg())
    assert wc.n_sat == 66
    assert abs(wc.period - 95 * 60) < 5 * 60  # 约 95 min ±5min


def test_position_ranges():
    wc = WalkerConstellation(_cfg())
    pos = wc.all_positions(0)
    assert pos.shape == (66, 3)
    assert pos[:, 0].min() >= -180 and pos[:, 0].max() <= 180  # lon
    assert pos[:, 1].min() >= -90 and pos[:, 1].max() <= 90    # lat
    assert np.allclose(pos[:, 2], 550.0)  # alt


def test_lat_bounded_by_inclination():
    wc = WalkerConstellation(_cfg())
    # 倾角 53°，卫星纬度应在 [-53, 53]
    for t in [0, 1000, 5000]:
        pos = wc.all_positions(t)
        assert pos[:, 1].max() <= 53 + 1
        assert pos[:, 1].min() >= -53 - 1


def test_orbit_periodicity():
    wc = WalkerConstellation(_cfg())
    p0 = wc.all_positions(0)
    p1 = wc.all_positions(wc.period)
    # 一个周期后回到起点（J2 RAAN 漂移不影响同一卫星的纬度幅角周期）
    # 经度因 RAAN 漂移与地球自转会有小偏差，主要看纬度
    assert np.allclose(p0[:, 1], p1[:, 1], atol=0.5)  # 纬度


def test_same_plane_phase():
    wc = WalkerConstellation(_cfg())
    # 同面相邻卫星纬度幅角差 = 360/11
    sat_phase = 360 / 11
    # 在 t=0，同面卫星 u0 差 = sat_phase
    u0 = wc.u0.reshape(wc.planes, wc.sats_per_plane)
    diff = np.rad2deg(u0[0, 1] - u0[0, 0])
    assert abs(diff - sat_phase) < 1e-6 or abs(diff + (360 - sat_phase)) < 1e-6


def test_reproducible():
    a = WalkerConstellation(_cfg()).all_positions(12345.0)
    b = WalkerConstellation(_cfg()).all_positions(12345.0)
    assert np.allclose(a, b)
