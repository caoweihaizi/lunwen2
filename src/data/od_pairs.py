"""混合 OD 配对 → 卫星 commodity 聚合（完全向量化）。

三股：随机 / 重力 / 热点。每条 OD 标 flow_type。
按目的卫星聚合为 commodity。用 np.add.at 向量化，避免 Python 循环。

比例正确性：每股总流量 = π_x × 全网总需求（严格守恒）。
- 随机股：每桶 D_s*π_rand 分到 1 个随机目的桶。
- 重力股：每桶 D_s*π_grav 按 grav_p[s,:] 全分配到所有目的桶（不截断）。
- 热点股：热点源桶的 D_s*π_hot 分到对应热点目的；非热点桶的 π_hot 并入随机股。
"""
from __future__ import annotations

import numpy as np

from .hotspots import select_hotspots


def _gravity_prob(bucket_weights, bucket_lon, bucket_lat, gamma):
    """重力分配概率 P[d|s] (nb,nb)。P(d|s) ∝ W_d/dist(s,d)^gamma，行归一化。"""
    nb = len(bucket_weights)
    dlon = np.deg2rad(bucket_lon[:, None] - bucket_lon[None, :])
    a = (np.sin((np.deg2rad(bucket_lat[:, None] - bucket_lat[None, :])) / 2) ** 2
         + np.cos(np.deg2rad(bucket_lat[:, None])) * np.cos(np.deg2rad(bucket_lat[None, :]))
         * np.sin(dlon / 2) ** 2)
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dist = np.maximum(dist, 1.0)
    grav = bucket_weights[None, :] / (dist ** gamma)
    np.fill_diagonal(grav, 0.0)
    rowsum = grav.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum > 0, rowsum, 1.0)
    return grav / rowsum


def build_od_commodity(base_demand, positions, times, wp, cfg, rng):
    """混合 OD 配对 → 按目的卫星聚合的 commodity（向量化）。"""
    from src.topo.coverage import assign_primary_sat

    om = cfg.demand.od_mix
    pi_rand = float(om.random)
    pi_grav = float(om.gravity)
    pi_hot = float(om.hotspot)
    gamma = float(cfg.demand.od_gravity_gamma)
    K_hot = int(cfg.demand.od_n_hotspots)

    bdt = base_demand["bucket_demand_ts"]  # (T, nb)
    bw = base_demand["bucket_weights"]
    blon = base_demand["bucket_lon"]
    blat = base_demand["bucket_lat"]
    nb = base_demand["n_buckets"]
    T = bdt.shape[0]
    n_sat = int(cfg.constellation.n_sat)

    grav_p = _gravity_prob(bw, blon, blat, gamma)  # (nb,nb)
    rand_p = bw / max(bw.sum(), 1e-12)

    hot_pairs, _ = select_hotspots(wp, K=K_hot)
    hot_src = np.array([int(p[0]) for p in hot_pairs], dtype=np.int32)
    hot_dst = np.array([int(p[1]) for p in hot_pairs], dtype=np.int32)
    n_hot = len(hot_pairs)

    # 每时隙桶→主服务星
    bucket_to_sat = np.full((T, nb), -1, dtype=np.int16)
    for t in range(T):
        sat, _ = assign_primary_sat(positions[t], blon, blat, cfg)
        bucket_to_sat[t] = sat

    # 随机目的（T, nb）
    rand_dest = np.empty((T, nb), dtype=np.int32)
    for t in range(T):
        rand_dest[t] = rng.choice(nb, size=nb, p=rand_p)

    # 三股严格按 π 比例切全网需求：
    # 随机股源需求 = demand_t * π_rand（每桶按其需求占比）
    # 重力股源需求 = demand_t * π_grav
    # 热点股 = Σ demand_t * π_hot，均分 n_hot 对（独立于源桶，从热点源卫星发往目的卫星）
    d_rand = bdt * pi_rand          # (T,nb)
    d_grav = bdt * pi_grav          # (T,nb)

    commodity_ts = []
    ft_counts = np.zeros(3, dtype=np.float64)

    for t in range(T):
        b2s = bucket_to_sat[t]
        valid_sat = b2s >= 0

        # --- 随机股 ---
        dst_b = rand_dest[t]
        src_s = b2s
        dst_s = b2s[dst_b]
        m = valid_sat & (dst_s >= 0) & (d_rand[t] > 0)
        rand_rows = np.zeros((0, 4))
        if m.any():
            ss = src_s[m].astype(np.int64)
            ds = dst_s[m].astype(np.int64)
            vv = d_rand[t][m]
            flat = ss * (n_sat + 1) + ds
            uniq, inv = np.unique(flat, return_inverse=True)
            acc = np.zeros(len(uniq), dtype=np.float64)
            np.add.at(acc, inv, vv)
            rand_rows = np.stack(
                [uniq // (n_sat + 1), uniq % (n_sat + 1), acc, np.zeros(len(uniq))], axis=1
            )
            ft_counts[0] += acc.sum()

        # --- 重力股（全分配）---
        gd = d_grav[t][:, None] * grav_p  # (nb,nb)
        ss2, ds2 = np.where(gd > 0)
        grav_rows = np.zeros((0, 4))
        if len(ss2):
            vv2 = gd[ss2, ds2]
            src_sat2 = b2s[ss2]
            dst_sat2 = b2s[ds2]
            m2 = (src_sat2 >= 0) & (dst_sat2 >= 0) & (vv2 > 0)
            if m2.any():
                ss2 = src_sat2[m2].astype(np.int64)
                ds2 = dst_sat2[m2].astype(np.int64)
                vv2 = vv2[m2]
                flat2 = ss2 * (n_sat + 1) + ds2
                uniq2, inv2 = np.unique(flat2, return_inverse=True)
                acc2 = np.zeros(len(uniq2), dtype=np.float64)
                np.add.at(acc2, inv2, vv2)
                grav_rows = np.stack(
                    [uniq2 // (n_sat + 1), uniq2 % (n_sat + 1), acc2, np.ones(len(uniq2))], axis=1
                )
                ft_counts[1] += acc2.sum()

        # --- 热点股：全网 π_hot 份额均分 n_hot 对 ---
        D_total = float(bdt[t].sum())
        per_pair = D_total * pi_hot / max(n_hot, 1)
        hot_rows = np.zeros((0, 4))
        if per_pair > 0:
            src_sat3 = b2s[hot_src]
            dst_sat3 = b2s[hot_dst]
            m3 = (src_sat3 >= 0) & (dst_sat3 >= 0)
            if m3.any():
                ss3 = src_sat3[m3].astype(np.int64)
                ds3 = dst_sat3[m3].astype(np.int64)
                vv3 = np.full(m3.sum(), per_pair, dtype=np.float64)
                flat3 = ss3 * (n_sat + 1) + ds3
                uniq3, inv3 = np.unique(flat3, return_inverse=True)
                acc3 = np.zeros(len(uniq3), dtype=np.float64)
                np.add.at(acc3, inv3, vv3)
                hot_rows = np.stack(
                    [uniq3 // (n_sat + 1), uniq3 % (n_sat + 1), acc3, np.full(len(uniq3), 2)], axis=1
                )
                ft_counts[2] += acc3.sum()

        arr = np.concatenate([rand_rows, grav_rows, hot_rows], axis=0)
        commodity_ts.append(arr)

    return {
        "commodity_ts": commodity_ts,
        "bucket_to_sat": bucket_to_sat,
        "flow_type_counts": ft_counts,
        "hotspot_pairs": hot_pairs,
    }
