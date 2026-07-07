"""P2 阶段入口：星座拓扑与覆盖生成。

流程：
1. 构造 WalkerConstellation
2. 读 P1 resampled_times，生成对齐的拓扑时间序列（稀疏边表）
3. 覆盖映射：采样时隙验证覆盖率
4. 势函数：代表性目的样本
5. 落盘 data/interim/topo/ + 更新 DATA_DICT
6. ExperimentRecord 记录
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
    resolve_paths,
    seed_everything,
)
from src.data import io  # noqa: E402
from src.topo import (  # noqa: E402
    WalkerConstellation,
    assign_primary_sat,
    coverage_half_angle,
    generate_topology_series,
    potential_propagation_delay,
    potential_shortest_hops,
    satellite_coverage,
)
from src.topo.topology import build_planned_topology  # noqa: E402


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p2_main", "P2")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P2")
    paths = resolve_paths(cfg)
    interim = paths["data_interim"]
    topo_dir = interim / "topo"
    topo_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== P2 星座拓扑与覆盖生成 ===")
    wc = WalkerConstellation(cfg)
    log.info(f"星座: {wc.n_sat}星 Walker {cfg.constellation.walker.planes}/"
             f"{wc.n_sat}/{cfg.constellation.walker.f} 倾角{cfg.constellation.walker.inclination_deg}° "
             f"高度{cfg.constellation.walker.altitude_km}km 周期{wc.period/60:.1f}min")

    # 读 P1 resampled_times 对齐
    geant_times = io.load_pickle(interim / "geant" / "resampled_times.pkl")["times"]
    log.info(f"与 P1 对齐时隙数: {len(geant_times)} ({geant_times[0]} ~ {geant_times[-1]})")

    # 生成拓扑时间序列
    log.info("生成拓扑时间序列（稀疏边表）...")
    series = generate_topology_series(cfg, geant_times)
    edge_counts = np.array([len(e) for e in series["edge_lists"]])
    log.info(f"拓扑序列完成: positions{series['positions'].shape} "
             f"边数 mean{edge_counts.mean():.0f} min{edge_counts.min()} max{edge_counts.max()}")

    # 落盘（边表存为 object npz）
    io.save_npz(topo_dir / "positions.npz",
                positions=series["positions"])
    io.save_pickle({"times": geant_times, "sat_ids": series["sat_ids"]},
                   topo_dir / "times.pkl")
    # 边表序列：存为单个 npz，每时隙边数不同，用 object 数组
    edge_obj = np.array(series["edge_lists"], dtype=object)
    dist_obj = np.array(series["edge_dists"], dtype=object)
    delay_obj = np.array(series["edge_delays"], dtype=object)
    np.savez_compressed(topo_dir / "topo_series.npz",
                        edge_lists=edge_obj, edge_dists=dist_obj, edge_delays=delay_obj)

    # 覆盖验证（采样几个时隙 + 用 WorldPop 有效像元采样）
    log.info("覆盖映射验证...")
    wp = io.load_npz(interim / "worldpop" / "weights.npz")
    # 为快速验证，按权重采样 20000 个有效像元
    rng = np.random.RandomState(0)
    K = len(wp["weights"])
    idx = rng.choice(K, size=min(20000, K), replace=False)
    glo = wp["lon"][idx]; gla = wp["lat"][idx]
    sample_times = [0, len(geant_times) // 4, len(geant_times) // 2, len(geant_times) - 1]
    coverage_rates = []
    for ti in sample_times:
        pos = series["positions"][ti]
        cov = np.zeros(len(glo), dtype=bool)
        for s in range(wc.n_sat):
            cov |= satellite_coverage(pos[s], glo, gla, cfg)
        coverage_rates.append(float(cov.mean()))
    log.info(f"采样覆盖率: {[round(r,3) for r in coverage_rates]}")

    # 势函数样本：6 个均匀分布的目的卫星，在 t=0 预计算
    log.info("势函数样本预计算...")
    pos0 = series["positions"][0]
    topo0 = build_planned_topology(pos0, cfg)
    sample_dsts = [0, 11, 22, 33, 44, 55]  # 每面第 1 颗
    pot_hops = {}
    pot_delay = {}
    for d in sample_dsts:
        pot_hops[d] = potential_shortest_hops(topo0["adj"], d)
        pot_delay[d] = potential_propagation_delay(topo0["prop_delay_ms"], d)
    io.save_npz(topo_dir / "potential_sample.npz",
                **{f"hops_dst{d}": pot_hops[d] for d in sample_dsts},
                **{f"delay_dst{d}": pot_delay[d] for d in sample_dsts},
                sample_dsts=np.array(sample_dsts))
    max_hops = max(int(p.max()) for p in pot_hops.values())
    log.info(f"势函数样本: {len(sample_dsts)} 目的, 最远{max_hops}跳")

    # 链路时延统计（t=0）
    d0 = topo0["prop_delay_ms"][topo0["adj"] > 0]
    log.info(f"t=0 链路时延 ms: min{d0.min():.2f} max{d0.max():.2f} mean{d0.mean():.2f}")

    # meta（全部转为 python 原生类型，避免 yaml/numpy 冲突）
    def _native(x):
        if isinstance(x, dict):
            return {k: _native(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_native(v) for v in x]
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return _native(x.tolist())
        return x

    meta = {
        "n_sat": wc.n_sat,
        "walker": {
            "planes": int(cfg.constellation.walker.planes),
            "sats_per_plane": int(cfg.constellation.walker.sats_per_plane),
            "f": int(cfg.constellation.walker.f),
            "inclination_deg": float(cfg.constellation.walker.inclination_deg),
            "altitude_km": float(cfg.constellation.walker.altitude_km),
        },
        "period_min": wc.period / 60,
        "min_elevation_deg": float(cfg.constellation.min_elevation_deg),
        "link_capacity_gbps": float(cfg.constellation.link_capacity_gbps),
        "max_isl_distance_km": float(cfg.constellation.max_isl_distance_km),
        "n_timeslots": len(geant_times),
        "time_start": geant_times[0].isoformat(),
        "time_end": geant_times[-1].isoformat(),
        "edges_mean": float(edge_counts.mean()),
        "edges_min": int(edge_counts.min()),
        "edges_max": int(edge_counts.max()),
        "coverage_rate_samples": coverage_rates,
        "coverage_half_angle_deg": float(np.rad2deg(coverage_half_angle(wc.alt, cfg.constellation.min_elevation_deg))),
        "potential_max_hops": max_hops,
        "prop_delay_ms_min": float(d0.min()),
        "prop_delay_ms_max": float(d0.max()),
        "interim_files": ["topo/positions.npz", "topo/times.pkl",
                          "topo/topo_series.npz", "topo/potential_sample.npz"],
        "note_coverage": "53°倾角星座覆盖±53°带(97.6%人口), 高纬无覆盖属预期; 5°仰角达96%",
    }
    with open(topo_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 指标
    rec.log_metric("n_sat", wc.n_sat)
    rec.log_metric("n_timeslots", len(geant_times))
    rec.log_metric("edges_mean", float(edge_counts.mean()))
    rec.log_metric("coverage_rate_t0", coverage_rates[0])
    rec.log_metric("potential_max_hops", max_hops)
    rec.log_metric("prop_delay_ms_min", float(d0.min()))
    rec.log_metric("prop_delay_ms_max", float(d0.max()))

    # 更新 DATA_DICT（追加 constellation 段）
    import yaml
    dd_path = interim / "DATA_DICT.yaml"
    if dd_path.exists():
        with open(dd_path, "r", encoding="utf-8") as f:
            dd = yaml.safe_load(f) or {}
    else:
        dd = {}
    dd["constellation"] = _native(meta)
    with open(dd_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dd, f, allow_unicode=True, sort_keys=False)
    log.info(f"DATA_DICT 更新 constellation 段")

    rec.log_output(str(topo_dir))
    out = rec.finish(status="success")
    log.info(f"P2 完成，记录落盘: {out}")
    log.info("P2 tracking OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
