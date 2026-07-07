"""覆盖与势函数单元测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config
from src.topo.orbit import WalkerConstellation
from src.topo.topology import build_planned_topology
from src.topo.coverage import satellite_coverage, assign_primary_sat, coverage_half_angle
from src.topo.potential import potential_shortest_hops


def _cfg():
    return load_config(cli=False)


def test_coverage_half_angle_reasonable():
    # 550km, 10度仰角 → 覆盖地心角约 12-16 度（半径约 1300-1700km）
    theta = coverage_half_angle(550.0, 10.0)
    deg = np.rad2deg(theta)
    assert 10 < deg < 18


def test_satellite_overhead_full_coverage():
    """卫星正下方的点必被覆盖。"""
    cfg = _cfg()
    sat_pos = (10.0, 20.0, 550.0)
    lon = np.array([10.0])
    lat = np.array([20.0])
    cov = satellite_coverage(sat_pos, lon, lat, cfg)
    assert cov.all()


def test_no_blind_spot_at_t0():
    """53° 倾角星座覆盖 ±53° 内区域（97.6% 人口）。高纬无覆盖属预期。
    采样 ±50° 内网格，覆盖率应 >0.95。"""
    cfg = _cfg()
    wc = WalkerConstellation(cfg)
    pos = wc.all_positions(0)
    lons = np.linspace(-179, 179, 70)
    lats = np.linspace(-50, 50, 70)
    LO, LA = np.meshgrid(lons, lats)
    glo, gla = LO.ravel(), LA.ravel()
    covered = np.zeros(len(glo), dtype=bool)
    for s in range(66):
        covered |= satellite_coverage(pos[s], glo, gla, cfg)
    assert covered.mean() > 0.95, f"±50°内覆盖率 {covered.mean():.3f} 过低"


def test_assign_primary_sat_returns_valid():
    cfg = _cfg()
    wc = WalkerConstellation(cfg)
    pos = wc.all_positions(0)
    # 选中低纬点，确保有覆盖
    glo = np.array([0.0, 30.0, -40.0])
    gla = np.array([0.0, 20.0, -10.0])
    sat, el = assign_primary_sat(pos, glo, gla, cfg)
    assert (sat >= 0).all()
    assert (el > 5).all()  # 主服务星仰角应 > 最小仰角(5°)


def test_potential_shortest_hops():
    cfg = _cfg()
    wc = WalkerConstellation(cfg)
    pos = wc.all_positions(0)
    topo = build_planned_topology(pos, cfg)
    phi = potential_shortest_hops(topo["adj"], dst=0)
    assert phi[0] == 0
    # 邻居势值 1
    nbrs = np.where(topo["adj"][0])[0]
    assert (phi[nbrs] == 1).all()
    # 最远不超过 8 跳
    assert phi.max() <= 8
