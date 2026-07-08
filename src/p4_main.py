"""P4 阶段入口：离散事件仿真器与离线策略（流式落盘版）。

流式：每 500 时隙 flush link_state 到磁盘分片，内存恒定。
默认不存明细表（keep_detail=False），仅 link_state + summary。
明细表审计时用单独脚本抽样生成。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    ExperimentRecord,
    get_logger,
    load_config,
    make_seed_stream,
    resolve_paths,
    seed_everything,
)
from src.data import io  # noqa: E402
from src.sim import (  # noqa: E402
    FlowLevelSimulator,
    DijkstraPolicy,
    ECMPPolicy,
    QueueAwareStochasticPolicy,
)
from src.sim.refine_k0 import refine_k0  # noqa: E402
from src.sim.failure import inject_failures  # noqa: E402


def _scale_commodity(cb, factor):
    return [np.column_stack([a[:, 0], a[:, 1], a[:, 2] * factor, a[:, 3]])
            if len(a) else a for a in cb]


def _make_flush_cb(run_dir):
    """返回 flush_callback：把分片落盘为 link_state_shardN.npz。"""
    def cb(shard_idx, ls, nc, lc, e2e, t_range):
        if ls:
            arr = np.array(ls, dtype=np.float64)  # (N, 10): t,i,j,offered,carried,queue,drop,util,delay,dist
            np.savez_compressed(run_dir / f"link_state_shard{shard_idx}.npz",
                                 link_state=arr, t_range=t_range)
        ls.clear()
    return cb


def _run_one(sim, cb, el, ed, de, times, policy, failed, rng, tag, log):
    log.info(f"  run {tag} ...")
    t0 = time.time()
    res = sim.run(cb["commodity"], el, ed, de, times, policy,
                  failed_edges=failed, rng=rng,
                  flush_callback=cb["flush"], flush_every=500, keep_detail=False)
    dt = time.time() - t0
    u = res["all_utils"]
    drop_rate = res["tot_drop"] / max(res["tot_offered"], 1e-9)
    log.info(f"    {dt:.0f}s | util中位{np.median(u):.3f} P95{np.percentile(u,95):.3f} "
             f"max{u.max():.2f} 丢包{drop_rate*100:.1f}% e2e{res['e2e_mean_ms']:.1f}ms")
    return res, {
        "util_median": float(np.median(u)),
        "util_p95": float(np.percentile(u, 95)),
        "util_max": float(np.max(u)),
        "drop_rate": float(drop_rate),
        "e2e_mean_ms": float(res["e2e_mean_ms"]),
        "n_slots": res["n_slots"],
        "n_shards": res["n_shards"],
        "tot_offered": float(res["tot_offered"]),
        "tot_carried": float(res["tot_carried"]),
        "tot_drop": float(res["tot_drop"]),
    }


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p4_main", "P4")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P4")
    paths = resolve_paths(cfg)
    out_dir = paths["data_processed"] / "sim"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== P4 离散事件仿真器与离线策略（流式）===")

    cb_base = np.load(paths["data_processed"] / "demand" / "commodity_baseline.npz",
                      allow_pickle=True)["commodity"]
    cb_burst = np.load(paths["data_processed"] / "demand" / "commodity_burst.npz",
                       allow_pickle=True)["commodity"]
    ts_npz = np.load(paths["data_interim"] / "topo" / "topo_series.npz", allow_pickle=True)
    el, ed, de = ts_npz["edge_lists"], ts_npz["edge_dists"], ts_npz["edge_delays"]
    times = io.load_pickle(paths["data_interim"] / "topo" / "times.pkl")["times"]
    log.info(f"读入: commodity{len(cb_base)}时隙 拓扑{len(el)}时隙")

    sim = FlowLevelSimulator(cfg, cfg.data.timeslot_minutes)

    # 1. refine k0（基于 k0=1 等效需求，返回绝对 k0）
    # P3 commodity 含 P3 标定时的 k0（记录在 demand/meta.json），先除掉得 k0=1 等效，
    # 避免 refine 时 k0 累加膨胀。不读 config.k0（可能已被上次 P4 回填）。
    import json as _json
    p3_meta = _json.load(open(paths["data_processed"] / "demand" / "meta.json"))
    p3_k0 = float(p3_meta["k0"])
    log.info(f"P3 commodity 含 k0={p3_k0:.4f}，除掉得 k0=1 等效用于 refine")
    cb_base_unit = _scale_commodity(cb_base, 1.0 / p3_k0)
    cb_burst_unit = _scale_commodity(cb_burst, 1.0 / p3_k0)
    log.info("refine k0 (ECMP, 目标 P95≈0.45, 基于 k0=1 等效需求)...")
    new_k0, k0_stats = refine_k0(cb_base_unit, el, ed, de, times, cfg, target_p95=0.45, sample_slots=1000)
    log.info(f"refine: 新k0={new_k0:.4f} P95={k0_stats['util_p95']:.3f} 丢包{k0_stats['drop_rate']*100:.1f}%")

    cb_base = _scale_commodity(cb_base_unit, new_k0)
    cb_burst = _scale_commodity(cb_burst_unit, new_k0)

    rng_fail = np.random.RandomState(next(make_seed_stream(cfg.seed.data, "failure")))
    failed_compound = inject_failures(el, cfg, rng_fail, "compound")
    log.info(f"故障(compound): {len(failed_compound)} 条链路全程失效")

    policies = [("dijkstra", DijkstraPolicy), ("ecmp", ECMPPolicy), ("queue_aware", QueueAwareStochasticPolicy)]
    scenarios = [("baseline", cb_base, set()), ("burst", cb_burst, set()), ("compound", cb_burst, failed_compound)]
    all_summaries = {}

    for pname, Pcls in policies:
        for sname, cb_scn, failed in scenarios:
            tag = f"{pname}_{sname}"
            run_dir = out_dir / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            pol = Pcls()
            rng_q = np.random.RandomState(next(make_seed_stream(cfg.seed.data, f"qaware_{tag}"))) if pname == "queue_aware" else None
            cb = {"commodity": cb_scn, "flush": _make_flush_cb(run_dir)}
            res, summ = _run_one(sim, cb, el, ed, de, times, pol, failed, rng_q, tag, log)
            all_summaries[tag] = summ
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summ, f, ensure_ascii=False, indent=2)

    # config 回填 k0
    log.info(f"config demand.k0 回填 = {new_k0:.4f}")
    _update_k0_in_config(paths["root"] / "configs" / "config.yaml", new_k0)

    # DATA_DICT
    import yaml
    dd_path = paths["data_interim"] / "DATA_DICT.yaml"
    with open(dd_path, "r", encoding="utf-8") as f:
        dd = yaml.safe_load(f) or {}
    dd["simulation"] = {
        "k0_refined": new_k0,
        "k0_refine_stats": k0_stats,
        "n_failed_edges_compound": len(failed_compound),
        "policies": [p for p, _ in policies],
        "scenarios": [s for s, _, _ in scenarios],
        "runs": all_summaries,
        "note": "流式落盘，每500时隙一shard；keep_detail=False，明细表审计时单独抽样",
    }
    with open(dd_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dd, f, allow_unicode=True, sort_keys=False)

    for tag, summ in all_summaries.items():
        rec.log_metric(f"{tag}_util_p95", summ["util_p95"])
        rec.log_metric(f"{tag}_drop_rate", summ["drop_rate"])
        rec.log_metric(f"{tag}_e2e_ms", summ["e2e_mean_ms"])
    rec.log_metric("k0_refined", new_k0)
    rec.log_metric("n_failed_edges", len(failed_compound))

    rec.log_output(str(out_dir))
    out = rec.finish(status="success")
    log.info(f"P4 完成，记录落盘: {out}")
    log.info("P4 tracking OK")
    return 0


def _update_k0_in_config(cfg_path: Path, new_k0: float):
    txt = cfg_path.read_text(encoding="utf-8")
    lines = txt.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("k0:"):
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}k0: {new_k0}      # P4 ECMP refine 回填（P95≈0.45，中负载）"
            break
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
