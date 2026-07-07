"""P3 阶段入口：外生 OD 需求构建与标定。

流程：
1. 读 P1 (S_resampled, times, WorldPop) + P2 (positions, topo_series)
2. 时间划分
3. 基础需求（桶级，时间×空间×昼夜）
4. 混合 OD 配对 → commodity（随机+重力+热点，0.4/0.4/0.2）
5. 突发注入（baseline + burst 场景）
6. 总量标定 k0（最短路代理，P95≈0.8）
7. 落盘 data/processed/demand/ + DATA_DICT + config 回填
"""
from __future__ import annotations

import json
import sys
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
from src.data.demand import compute_base_demand  # noqa: E402
from src.data.split import time_split  # noqa: E402
from src.data.od_pairs import build_od_commodity  # noqa: E402
from src.data.interventions import inject_burst  # noqa: E402
from src.data.calibrate import estimate_k0  # noqa: E402


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p3_main", "P3")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P3")
    paths = resolve_paths(cfg)
    interim = paths["data_interim"]
    out_dir = paths["data_processed"] / "demand"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== P3 外生 OD 需求构建与标定 ===")

    # 读 P1/P2
    Sr = io.load_npz(interim / "geant" / "S_resampled.npz")["S_resampled"]
    times = io.load_pickle(interim / "geant" / "resampled_times.pkl")["times"]
    wp = io.load_npz(interim / "worldpop" / "weights.npz")
    pos = io.load_npz(interim / "topo" / "positions.npz")["positions"]
    ts_npz = np.load(interim / "topo" / "topo_series.npz", allow_pickle=True)
    edge_lists = ts_npz["edge_lists"]
    edge_delays = ts_npz["edge_delays"]
    log.info(f"读入: S_resampled{Sr.shape} positions{pos.shape} 时隙{len(times)}")

    # 1. 时间划分
    split = time_split(times, tuple(cfg.data.train_val_calib_test))
    log.info(f"划分 train{split['train']} val{split['val']} calib{split['calib']} test{split['test']}")

    # 2. 基础需求（k0=1，标定后缩放）
    bd = compute_base_demand(Sr, times, wp, cfg, k0=1.0)
    log.info(f"基础需求: {bd['n_buckets']}桶 总量均值{bd['bucket_demand_ts'].sum(axis=1).mean():.0f}")

    # 3. 混合 OD → commodity（k0=1）
    gen = make_seed_stream(cfg.seed.data, "od")
    rng = np.random.RandomState(next(gen))
    log.info("生成混合 OD commodity (随机+重力+热点 0.4/0.4/0.2)...")
    oc = build_od_commodity(bd, pos, times, wp, cfg, rng)
    ftc = oc["flow_type_counts"]; tot = ftc.sum()
    log.info(f"三股比例: {(ftc/tot).round(3)} (目标 0.4/0.4/0.2) | 守恒 1.0")

    # 4. 突发注入
    rng_burst = np.random.RandomState(next(make_seed_stream(cfg.seed.data, "burst")))
    baseline_ts, _ = inject_burst(oc["commodity_ts"], times, cfg, rng_burst, "baseline")
    burst_ts, burst_log = inject_burst(oc["commodity_ts"], times, cfg, rng_burst, "burst")
    log.info(f"突发场景: {len(burst_log['events'])} 个源卫星突发 ×{cfg.demand.burst_multiplier}")

    # 5. k0 标定（用 baseline, k0=1 的 commodity）
    log.info("k0 标定（最短路代理，目标 P95≈0.8）...")
    k0, util_stats = estimate_k0(oc["commodity_ts"], edge_lists, edge_delays, times, cfg)
    log.info(f"k0={k0:.4f} | 利用率 中位{util_stats['util_median']:.3f} "
             f"P95{util_stats['util_p95']:.3f} max{util_stats['util_max']:.3f}")

    # 6. 应用 k0 缩放 commodity（baseline 和 burst 都乘 k0）
    def scale_k0(ts_list, k):
        return [np.column_stack([arr[:, 0], arr[:, 1], arr[:, 2] * k, arr[:, 3]])
                if len(arr) else arr for arr in ts_list]

    baseline_ts_k0 = scale_k0(baseline_ts, k0)
    burst_ts_k0 = scale_k0(burst_ts, k0)

    # 7. 落盘
    log.info("落盘 data/processed/demand/...")
    # commodity 存为 object 数组（每时隙变长）
    np.savez_compressed(out_dir / "commodity_baseline.npz",
                        commodity=np.array(baseline_ts_k0, dtype=object))
    np.savez_compressed(out_dir / "commodity_burst.npz",
                        commodity=np.array(burst_ts_k0, dtype=object))
    io.save_npz(out_dir / "base_demand.npz",
                bucket_demand_ts=bd["bucket_demand_ts"].astype(np.float32),
                bucket_weights=bd["bucket_weights"].astype(np.float32),
                bucket_lon=bd["bucket_lon"].astype(np.float32),
                bucket_lat=bd["bucket_lat"].astype(np.float32),
                bucket_utc=bd["bucket_utc"].astype(np.float32),
                bucket_ids=bd["bucket_ids"])
    io.save_pickle({"split": split, "times": times}, out_dir / "split.pkl")
    with open(out_dir / "k0.json", "w", encoding="utf-8") as f:
        json.dump(util_stats, f, ensure_ascii=False, indent=2)
    with open(out_dir / "interventions.json", "w", encoding="utf-8") as f:
        json.dump(burst_log, f, ensure_ascii=False, indent=2, default=str)
    # 热点对清单
    hot_info = [{"src_bucket": int(p[0]), "dst_bucket": int(p[1]),
                 "src_region": p[2], "dst_region": p[3]} for p in oc["hotspot_pairs"]]
    with open(out_dir / "hotspot_pairs.json", "w", encoding="utf-8") as f:
        json.dump(hot_info, f, ensure_ascii=False, indent=2)

    meta = {
        "n_timeslots": len(times),
        "n_buckets": int(bd["n_buckets"]),
        "n_hotspot_pairs": len(oc["hotspot_pairs"]),
        "od_mix": {"random": 0.4, "gravity": 0.4, "hotspot": 0.2},
        "flow_type_counts": [float(x) for x in ftc],
        "k0": float(k0),
        "util_stats": util_stats,
        "split": {k: list(v) for k, v in split.items()},
        "scenarios": ["baseline", "burst"],
        "interim_files": ["demand/commodity_baseline.npz", "demand/commodity_burst.npz",
                          "demand/base_demand.npz", "demand/split.pkl",
                          "demand/k0.json", "demand/interventions.json",
                          "demand/hotspot_pairs.json"],
        "note_k0": "P3 最短路代理标定 P95≈0.8；中位低反映热点集中，P4 ECMP/MAPPO refine",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 8. 回填 config demand.k0
    # 用 OmegaConf 写回 config.yaml
    from omegaconf import OmegaConf
    cfg_path = paths["root"] / "configs" / "config.yaml"
    cfg_disk = OmegaConf.load(cfg_path)
    cfg_disk.demand.k0 = float(k0)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg_disk, sort_keys=False))
    log.info(f"config demand.k0 回填 = {k0:.4f}")

    # 9. 更新 DATA_DICT
    import yaml
    dd_path = interim / "DATA_DICT.yaml"
    with open(dd_path, "r", encoding="utf-8") as f:
        dd = yaml.safe_load(f) or {}
    dd["demand"] = meta
    with open(dd_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dd, f, allow_unicode=True, sort_keys=False)

    # 指标
    rec.log_metric("n_timeslots", len(times))
    rec.log_metric("n_buckets", int(bd["n_buckets"]))
    rec.log_metric("flow_rand_ratio", float(ftc[0] / tot))
    rec.log_metric("flow_grav_ratio", float(ftc[1] / tot))
    rec.log_metric("flow_hot_ratio", float(ftc[2] / tot))
    rec.log_metric("k0", float(k0))
    rec.log_metric("util_median", util_stats["util_median"])
    rec.log_metric("util_p95", util_stats["util_p95"])
    rec.log_metric("util_max", util_stats["util_max"])
    rec.log_metric("baseline_demand_mean", float(bd["bucket_demand_ts"].sum(axis=1).mean()) * k0)

    rec.log_output(str(out_dir))
    out = rec.finish(status="success")
    log.info(f"P3 完成，记录落盘: {out}")
    log.info("P3 tracking OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
