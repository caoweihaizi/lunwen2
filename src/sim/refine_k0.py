"""用 ECMP 真实路由 refine k0（技术大纲 §4.1.4，P4 决策）。

P3 用最短路代理标定，P4 用 ECMP 真实分布 refine。
目标：P95≈0.45（中负载），max≈1.0，丢包率可观但不极端。
中位低（66% 链路空载）是 66 星稀疏流量特性，改用 P95 口径（记入 8.2）。
"""
from __future__ import annotations

import numpy as np

from .simulator import FlowLevelSimulator
from .policies import ECMPPolicy


def refine_k0(commodity_ts, edge_lists, edge_dists, edge_delays, times, cfg,
              target_p95=0.45, sample_slots=1000):
    """二分 k0 使 ECMP 真实分布 P95≈target_p95。

    commodity_ts: 当前已含旧 k0 的 commodity（会被缩放）。
    返回 (new_k0_relative, util_stats) —— new_k0_relative 是相对当前 commodity 的倍数。
    最终 k0 = 旧 k0 × new_k0_relative。
    """
    sim = FlowLevelSimulator(cfg, cfg.data.timeslot_minutes)
    n_slots = min(sample_slots, len(times))

    def util_at_factor(factor):
        cb_scaled = [np.column_stack([a[:, 0], a[:, 1], a[:, 2] * factor, a[:, 3]])
                     if len(a) else a for a in commodity_ts[:n_slots]]
        pol = ECMPPolicy()
        res = sim.run(cb_scaled, edge_lists[:n_slots], edge_dists[:n_slots],
                      edge_delays[:n_slots], times[:n_slots], pol,
                      failed_edges=set(), max_slots=n_slots,
                      flush_callback=None, keep_detail=False)
        u = res["all_utils"]
        drop_rate = res["tot_drop"] / max(res["tot_offered"], 1e-9)
        return u, drop_rate

    # 二分 factor：P95 随 factor 单调增
    lo, hi = 0.5, 8.0
    best = None
    for _ in range(15):
        mid = (lo + hi) / 2
        u, drop = util_at_factor(mid)
        p95 = np.percentile(u, 95) if len(u) else 0
        if p95 < target_p95:
            lo = mid
        else:
            hi = mid
        best = mid
    u_final, drop_final = util_at_factor(best)
    stats = {
        "factor": float(best),
        "target_p95": target_p95,
        "util_median": float(np.median(u_final)),
        "util_p95": float(np.percentile(u_final, 95)),
        "util_max": float(np.max(u_final)),
        "drop_rate": float(drop_final),
        "n_sample_slots": n_slots,
        "note": "ECMP 真实路由 refine；中位低因 66 星稀疏流量，改用 P95 口径（8.2 局限性）",
    }
    return best, stats
