"""因果顺序审计（§4.7 第4项 + §7.2 第6项）。

三类干预执行顺序：
- 需求突发先于路由生效
- 链路故障先于标签生成生效（compound 故障链路从运行拓扑移除）
- 遥测缺失只改观测不改 ground truth（P5 ls_truth == P4 link_state）
"""
from __future__ import annotations

import json
import numpy as np

from src.common import resolve_paths


def audit_burst_before_routing(cfg):
    """突发在路由前注入：突发源卫星的 commodity 在突发时段[40%,60%)显著升高。

    burst 只放大 3 个源卫星，全网总量被稀释不明显，故检查突发源卫星的流出量。
    """
    p = resolve_paths(cfg)
    cb_base = np.load(p["data_processed"] / "demand" / "commodity_baseline.npz",
                      allow_pickle=True)["commodity"]
    cb_burst = np.load(p["data_processed"] / "demand" / "commodity_burst.npz",
                       allow_pickle=True)["commodity"]
    interv = json.load(open(p["data_processed"] / "demand" / "interventions.json"))
    burst_sats = [e["src_sat"] for e in interv["events"]]
    T = len(cb_base)
    t0, t1 = int(T * 0.4), int(T * 0.6)

    def src_sat_outflow(cb, t0, t1):
        out = np.zeros(t1 - t0)
        for ti in range(t0, t1):
            a = cb[ti]
            if len(a):
                m = np.isin(a[:, 0].astype(int), burst_sats)
                out[ti - t0] = a[m, 2].sum() if m.any() else 0
        return out

    base_out = src_sat_outflow(cb_base, t0, t1)
    burst_out = src_sat_outflow(cb_burst, t0, t1)
    # 只在有流量的时隙检查 ratio（源卫星该时隙无流量则 0×4=0，无法体现突发）
    has_flow = base_out > 1e-6
    ratio = burst_out[has_flow] / np.maximum(base_out[has_flow], 1e-9)
    higher = (ratio > 2.0).mean() if len(ratio) else 0

    base_sum = np.array([a[:, 2].sum() if len(a) else 0 for a in cb_base[:t0]])
    burst_sum = np.array([a[:, 2].sum() if len(a) else 0 for a in cb_burst[:t0]])
    non_burst = np.abs(burst_sum - base_sum).mean() / np.maximum(base_sum.mean(), 1e-9)
    return {"pass": bool(higher > 0.9 and non_burst < 0.01),
            "burst_src_ratio_mean": float(ratio.mean()),
            "burst_period_higher_ratio": float(higher),
            "non_burst_rel_diff": float(non_burst),
            "burst_sats": burst_sats}


def audit_failure_before_label(cfg, n_shards=10):
    """compound 故障链路在运行拓扑中被移除（不出现在 link_state）。

    审计：compound link_state 出现的边数 < baseline（故障链路缺失），
    且 compound 丢包率 > baseline（故障加剧拥塞）。
    """
    p = resolve_paths(cfg)
    base_dir = p["data_processed"] / "sim" / "ecmp_baseline"
    comp_dir = p["data_processed"] / "sim" / "ecmp_compound"

    base_drop = json.load(open(base_dir / "summary.json"))["drop_rate"]
    comp_drop = json.load(open(comp_dir / "summary.json"))["drop_rate"]

    base_shards = sorted(base_dir.glob("link_state_shard*.npz"))
    idxs = np.linspace(0, len(base_shards) - 1, min(n_shards, len(base_shards))).astype(int)
    base_edges = set(); comp_edges = set()
    for i in idxs:
        for d, sset in [(base_dir, base_edges), (comp_dir, comp_edges)]:
            sp = d / f"link_state_shard{int(i)}.npz"
            if not sp.exists():
                continue
            arr = np.load(sp)["link_state"]
            for r in arr:
                sset.add(frozenset({int(r[1]), int(r[2])}))
    missing_in_comp = base_edges - comp_edges
    return {"pass": bool(comp_drop > base_drop and len(missing_in_comp) > 0),
            "n_edges_missing_in_compound": len(missing_in_comp),
            "comp_drop_rate": float(comp_drop), "base_drop_rate": float(base_drop),
            "n_base_edges_sample": len(base_edges),
            "n_comp_edges_sample": len(comp_edges)}


def audit_missing_unchanges_truth(cfg, n_shards=20):
    """P5 ls_truth == P4 link_state（逐 shard 抽样）。"""
    p = resolve_paths(cfg)
    sim_dir = p["data_processed"] / "sim"
    obs_dir = p["data_processed"] / "observed"
    n_match = 0; n_check = 0
    for run_dir in sorted(sim_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run = run_dir.name
        obs_run_dir = obs_dir / f"{run}_mcar20"
        if not obs_run_dir.exists():
            continue
        sim_shards = sorted(run_dir.glob("link_state_shard*.npz"))
        obs_shards = sorted(obs_run_dir.glob("observed_shard*.npz"))
        idxs = np.linspace(0, len(sim_shards) - 1, min(5, len(sim_shards))).astype(int)
        for i in idxs:
            ls_p4 = np.load(sim_shards[i])["link_state"]
            ls_p5 = np.load(obs_shards[i])["ls_truth"]
            if np.array_equal(ls_p4, ls_p5):
                n_match += 1
            n_check += 1
        if n_check >= n_shards:
            break
    return {"pass": n_match == n_check, "n_match": n_match, "n_check": n_check}
