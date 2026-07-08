"""缺失统计与可复现性审计（§4.7 第11-12项 + §7.2 第8项）。"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from src.common import resolve_paths


def audit_missing_stats(cfg):
    """汇总 9 run × 机制的缺失率、MAR 高低负载差。"""
    p = resolve_paths(cfg)
    obs_dir = p["data_processed"] / "observed"
    summaries = {}
    for f in sorted(obs_dir.glob("*_missing_summary.json")):
        run = f.stem.replace("_missing_summary", "")
        summaries[run] = json.load(open(f))
    # 抽 ecmp_baseline 检查所有机制达标
    eb = summaries.get("ecmp_baseline", {})
    ok = True
    checks = {}
    for m, target in [("mcar20", 0.20), ("mcar40", 0.40), ("mar20", 0.20), ("block10", 0.20)]:
        if m in eb:
            actual = eb[m]["mean_actual_rate"]
            checks[m] = {"actual": actual, "target": target,
                         "pass": abs(actual - target) < 0.05}
            if not checks[m]["pass"]:
                ok = False
    # MAR 高低负载
    mar = eb.get("mar20", {})
    mar_ok = (mar.get("mar_high_load_miss", 0) > mar.get("mar_low_load_miss", 1) * 2)
    return {"pass": bool(ok and mar_ok), "checks": checks,
            "mar_high": mar.get("mar_high_load_miss"),
            "mar_low": mar.get("mar_low_load_miss"),
            "n_runs": len(summaries)}


def audit_reproducibility(cfg):
    """可复现：抽样重跑 P4 一个 shard，对比哈希。"""
    import hashlib
    from src.sim import FlowLevelSimulator, DijkstraPolicy
    from src.common import seed_everything
    from src.data import io

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

    # 重跑前 500 时隙（1 shard）
    sim = FlowLevelSimulator(cfg, cfg.data.timeslot_minutes)
    pol = DijkstraPolicy()
    captured = []
    def cb_flush(shard_idx, ls, nc, lc, e2e, t_range):
        if ls:
            captured.append(np.array(ls, dtype=np.float64))
    sim.run(cb[:500], el[:500], ed[:500], de[:500], times[:500], pol,
            failed_edges=set(), flush_callback=cb_flush, flush_every=500, keep_detail=False)
    rerun_arr = captured[0] if captured else np.array([])
    # 对比 P4 落盘的 shard0
    p4_shard = np.load(p["data_processed"] / "sim" / "dijkstra_baseline" / "link_state_shard0.npz")["link_state"]
    if len(rerun_arr) != len(p4_shard):
        return {"pass": False, "reason": f"长度不符 {len(rerun_arr)} vs {len(p4_shard)}"}
    # 浮点累加顺序差异导致 ~1e-10 噪声，用容差比较
    diff = np.abs(rerun_arr - p4_shard)
    max_rel = float((diff / np.maximum(np.abs(p4_shard), 1e-9)).max())
    return {"pass": max_rel < 1e-6, "max_rel_err": max_rel,
            "note": "浮点累加顺序噪声~1e-10，用容差1e-6判定可复现"}
