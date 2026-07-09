"""用 ECMP 真实路由 refine k0（技术大纲 §4.1.4，P4 决策）。

264 星下流量分布极度右偏，util max 恒≤1.0（服务被容量 cap），
改用 drop_rate 做目标：target_drop≈0.08（8% 丢包，有拥塞但不崩溃）。
"""
from __future__ import annotations

import numpy as np

from .simulator import FlowLevelSimulator
from .policies import ECMPPolicy


def refine_k0(commodity_ts_unit, edge_lists, edge_dists, edge_delays, times, cfg,
              target_drop=0.08, sample_slots=1000):
    """二分 k0 使 ECMP 真实分布 drop_rate≈target_drop。

    drop_rate 随 k0 单调增（k0 大→流量大→丢包多）。
    """
    sim = FlowLevelSimulator(cfg, cfg.data.timeslot_minutes)
    n_slots = min(sample_slots, len(times))

    def util_at_k0(k0):
        cb_scaled = [np.column_stack([a[:, 0], a[:, 1], a[:, 2] * k0, a[:, 3]])
                     if len(a) else a for a in commodity_ts_unit[:n_slots]]
        pol = ECMPPolicy()
        res = sim.run(cb_scaled, edge_lists[:n_slots], edge_dists[:n_slots],
                      edge_delays[:n_slots], times[:n_slots], pol,
                      failed_edges=set(), max_slots=n_slots,
                      flush_callback=None, keep_detail=False)
        u = res["all_utils"]
        drop_rate = res["tot_drop"] / max(res["tot_offered"], 1e-9)
        return u, drop_rate

    # 二分绝对 k0：drop_rate 随 k0 单调增
    lo, hi = 0.1, 30.0
    best = None
    for _ in range(20):
        mid = (lo + hi) / 2
        u, drop = util_at_k0(mid)
        if drop < target_drop:
            lo = mid
        else:
            hi = mid
        best = mid
    u_final, drop_final = util_at_k0(best)
    stats = {
        "k0": float(best),
        "target_drop": target_drop,
        "util_median": float(np.median(u_final)),
        "util_p95": float(np.percentile(u_final, 95)),
        "util_max": float(np.max(u_final)),
        "drop_rate": float(drop_final),
        "n_sample_slots": n_slots,
        "note": "ECMP refine，drop_rate 口径（util max 被 cap 无意义）",
    }
    return best, stats

