"""需求层干预：突发/热点/漂移注入（技术大纲 §4.2 步骤 5，§4.5.1）。

干预在路由前完成（P3），干预后的需求才是 P4 输入。
"""
from __future__ import annotations

import numpy as np


def inject_burst(commodity_ts, times, cfg, rng, scenario="burst"):
    """在 commodity 上注入突发。

    scenario:
      'baseline' — 无干预，原样返回
      'burst'    — 选若干源卫星，在指定时段需求放大 burst_multiplier 倍
      'compound' — burst + （故障在 P4 注入，P3 只做 demand 侧 burst）

    返回 (commodity_ts_scn, intervention_log)
    """
    mult = float(cfg.demand.burst_multiplier)
    T = len(times)
    n_sat = int(cfg.constellation.n_sat)

    if scenario == "baseline":
        return [arr.copy() for arr in commodity_ts], {
            "scenario": "baseline", "events": []
        }

    # burst：选 3 个源卫星，在 T 的中间 20% 时段放大
    n_burst_sat = 3
    burst_sats = rng.choice(n_sat, size=n_burst_sat, replace=False)
    t_start = int(T * 0.4)
    t_end = int(T * 0.6)
    events = []
    for s in burst_sats:
        events.append({
            "type": "burst", "src_sat": int(s),
            "t_start_idx": t_start, "t_end_idx": t_end,
            "multiplier": mult,
            "t_start": times[t_start].isoformat(),
            "t_end": times[t_end - 1].isoformat(),
        })

    scn_ts = []
    for t, arr in enumerate(commodity_ts):
        if t_start <= t < t_end and len(arr):
            arr2 = arr.copy()
            src = arr2[:, 0].astype(int)
            mask = np.isin(src, burst_sats)
            arr2[mask, 2] *= mult
            scn_ts.append(arr2)
        else:
            scn_ts.append(arr.copy())
    return scn_ts, {"scenario": scenario, "events": events}
