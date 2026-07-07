"""拓扑模块单元测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config
from src.topo.orbit import WalkerConstellation
from src.topo.topology import build_planned_topology


def _topo():
    cfg = load_config(cli=False)
    wc = WalkerConstellation(cfg)
    pos = wc.all_positions(0)
    return cfg, build_planned_topology(pos, cfg)


def test_degree_at_most_4():
    cfg, topo = _topo()
    deg = topo["adj"].sum(axis=1)
    assert deg.max() <= 4, f"max degree {deg.max()} > 4"


def test_adj_symmetric():
    _, topo = _topo()
    assert np.array_equal(topo["adj"], topo["adj"].T)


def test_intra_plane_neighbors_permanent():
    cfg, topo = _topo()
    planes = cfg.constellation.walker.planes
    spp = cfg.constellation.walker.sats_per_plane
    adj = topo["adj"]
    # 同面 ±1 必相连
    for p in range(planes):
        for s in range(spp):
            i = p * spp + s
            nxt = p * spp + (s + 1) % spp
            prv = p * spp + (s - 1) % spp
            assert adj[i, nxt], f"sat {i} 应连同面后继 {nxt}"
            assert adj[i, prv], f"sat {i} 应连同面前驱 {prv}"


def test_link_distance_positive():
    _, topo = _topo()
    adj = topo["adj"]
    d = topo["distance_km"][adj]
    assert (d > 0).all()
    assert d.min() > 100  # ISL 不会太近
    assert d.max() < 5000  # 受 max_isl_distance 约束


def test_prop_delay_reasonable():
    _, topo = _topo()
    adj = topo["adj"]
    dl = topo["prop_delay_ms"][adj]
    assert (dl > 0).all()
    assert dl.min() > 0.5 and dl.max() < 30  # ISL 典型 1-20 ms
