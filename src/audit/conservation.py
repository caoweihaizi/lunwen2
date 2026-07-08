"""守恒审计（§4.7 前3项 + §7.2 第5项）。

1. OD/commodity 守恒：Σ commodity == S(t)×k0
2. 链路 offered == carried + drop + queue
3. carried_load == Σ commodity_load（抽样 keep_detail 重跑）
"""
from __future__ import annotations

import numpy as np

from src.common import resolve_paths


def audit_od_conservation(cfg, tol=1e-3):
    """Σ commodity volume == base_demand 总量（P3 三股守恒）。

    commodity 与 base_demand 都含 P3 的 k0。对比两者时序总和（三股分配守恒）。
    """
    p = resolve_paths(cfg)
    cb = np.load(p["data_processed"] / "demand" / "commodity_baseline.npz",
                 allow_pickle=True)["commodity"]
    bd = np.load(p["data_processed"] / "demand" / "base_demand.npz")
    bucket_demand_ts = bd["bucket_demand_ts"]  # (T, n_buckets)，k0=1（P3 落盘 base_demand 用 k0=1）
    import json
    p3_k0 = float(json.load(open(p["data_processed"] / "demand" / "meta.json"))["k0"])
    p4_k0 = float(cfg.demand.k0)
    # commodity 落盘含 p3_k0；base_demand 落盘是 k0=1。所以 commodity == base × p3_k0
    comm_sum = np.array([a[:, 2].sum() if len(a) else 0 for a in cb])
    base_sum = bucket_demand_ts.sum(axis=1) * p3_k0
    diff = np.abs(comm_sum - base_sum)
    rel = diff / (np.abs(base_sum) + 1e-9)
    ok = bool((rel < tol).mean() > 0.99)
    return {"pass": ok, "max_rel_err": float(rel.max()),
            "mean_rel_err": float(rel.mean()), "p3_k0": p3_k0, "p4_k0": p4_k0,
            "note": "commodity 总量 == base_demand 总量（三股守恒），均含 P4 factor"}


def audit_link_conservation(cfg, sample_shards=10):
    """每 shard offered == carried+drop+queue（P4）。抽样若干 shard。"""
    p = resolve_paths(cfg)
    import glob
    max_diff = 0.0
    n_checked = 0
    for run_dir in sorted((p["data_processed"] / "sim").iterdir()):
        shards = sorted(run_dir.glob("link_state_shard*.npz"))
        # 抽样
        idxs = np.linspace(0, len(shards) - 1, min(sample_shards, len(shards))).astype(int)
        for i in idxs:
            ls = np.load(shards[i])["link_state"]
            off = ls[:, 3]; car = ls[:, 4]; dr = ls[:, 6]; q = ls[:, 5]
            diff = np.abs(off - (car + dr + q))
            max_diff = max(max_diff, float(diff.max()))
            n_checked += 1
    return {"pass": max_diff < 1e-3, "max_abs_err": max_diff, "n_shards_checked": n_checked}


def audit_commodity_load_restore(cfg, sample_slots=10):
    """carried_load == Σ commodity_load：抽样 keep_detail 重跑，验证明细可还原链路到达量。

    用 flush_callback 收集 link_state + link_commodity，
    对比 link_commodity 聚合的 per-edge load vs link_state offered。
    """
    from src.sim import FlowLevelSimulator, DijkstraPolicy
    from src.common import seed_everything
    from src.data import io
    import json

    p = resolve_paths(cfg)
    seed_everything(cfg.seed.data)
    cb = np.load(p["data_processed"] / "demand" / "commodity_baseline.npz",
                 allow_pickle=True)["commodity"]
    ts = np.load(p["data_interim"] / "topo" / "topo_series.npz", allow_pickle=True)
    el, ed, de = ts["edge_lists"], ts["edge_dists"], ts["edge_delays"]
    times = io.load_pickle(p["data_interim"] / "topo" / "times.pkl")["times"]

    p3_k0 = float(json.load(open(p["data_processed"] / "demand" / "meta.json"))["k0"])
    factor = float(cfg.demand.k0) / p3_k0
    cb = [np.column_stack([a[:, 0], a[:, 1], a[:, 2] * factor, a[:, 3]])
          if len(a) else a for a in cb]

    T = len(times)
    slot_idxs = np.linspace(0, T - 1, sample_slots).astype(int)
    sim = FlowLevelSimulator(cfg, cfg.data.timeslot_minutes)
    pol = DijkstraPolicy()

    collected = {"ls": [], "lc": []}

    def cb_flush(shard_idx, ls, nc, lc, e2e, t_range):
        if ls:
            collected["ls"].append(np.array(ls))
        if lc:
            collected["lc"].append(np.array(lc))

    # 跑覆盖抽样时隙的连续窗口（一次跑前 sample_slots 个时隙）
    t0 = 0; t1 = min(T, sample_slots + 2)
    sim.run(cb[t0:t1], el[t0:t1], ed[t0:t1], de[t0:t1], times[t0:t1],
            pol, failed_edges=set(), flush_callback=cb_flush,
            flush_every=sample_slots, keep_detail=True)

    if not collected["ls"] or not collected["lc"]:
        return {"pass": False, "note": "未收集到明细"}

    ls_arr = np.concatenate(collected["ls"])  # (N,10)
    lc_arr = np.concatenate(collected["lc"])  # (M,5): t,i,j,d,load
    # link_commodity 聚合 per (t,i,j)
    from collections import defaultdict
    edge_t_load = defaultdict(float)
    for row in lc_arr:
        key = (int(row[0]), int(row[1]), int(row[2]))
        edge_t_load[key] += float(row[4])
    # 对比 link_state offered（列3）
    diffs = []
    for r in ls_arr:
        key = (int(r[0]), int(r[1]), int(r[2]))
        off = float(r[3])
        agg = edge_t_load.get(key, 0.0)
        if off > 0 or agg > 0:
            diffs.append(abs(off - agg) / max(abs(off), 1e-9))
    max_rel = max(diffs) if diffs else 0.0
    return {"pass": max_rel < 0.01, "max_rel_err": float(max_rel),
            "n_edges_checked": len(diffs),
            "note": "keep_detail 抽样重跑，link_commodity 聚合 == link_state offered"}

