"""P1 阶段入口：外部数据接入与预处理。

流程：
1. 加载 GÉANT → 提取 S(t) → winsorize → 重采样到 3min → 落盘 + 诊断图
2. 加载 Abilene → 同上
3. 加载 WorldPop → 计算权重 → 落盘
4. 写数据字典 DATA_DICT.yaml
5. ExperimentRecord 记录指标
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
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
from src.data import (  # noqa: E402
    detect_time_gaps,
    extract_global_intensity,
    extract_od_relative_pattern,
    io,
    load_traffic_matrix_archive,
    load_worldpop,
    compute_region_weights,
    resample_to_timeslot,
    winsorize_intensity,
)
from src.data import diagnostics as diag  # noqa: E402
from src.common.config import PROJECT_ROOT  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _process_traffic(cfg, which: str, paths, log, rec) -> dict:
    """处理一个流量矩阵数据集，返回元信息。"""
    out_dir = paths["data_interim"] / which
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = paths["data_interim"] / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[{which}] 加载流量矩阵归档...")
    g = load_traffic_matrix_archive(cfg, which)
    n_nodes = len(g["nodes"])
    n_od = int((g["matrix"] > 0).any(axis=0).sum())  # 非零 OD 列数近似
    log.info(f"[{which}] shape={g['matrix'].shape} nodes={n_nodes} "
             f"gran={g['granularity_min']}min unit={g['unit']}")

    # 时间缺口
    gaps = detect_time_gaps(g["times"], g["granularity_min"])
    log.info(f"[{which}] 时间缺口 {len(gaps)} 处")
    for gp in gaps[:3]:
        log.info(f"      gap: {gp[0]} -> {gp[1]} ({gp[2]}min)")

    # S(t)
    S = extract_global_intensity(g["matrix"])
    S_clean, clip_stats = winsorize_intensity(S, percentile=99.9)
    log.info(f"[{which}] S(t) winsorize: {clip_stats['n_clipped']} 点截断 "
             f"max {clip_stats['orig_max']:.0f}->{clip_stats['clip_max']:.0f}")

    # OD 相对结构
    p_od = extract_od_relative_pattern(g["matrix"])

    # 重采样（用清洗后 S）
    Sr, tr = resample_to_timeslot(S_clean, g["times"], cfg.data.timeslot_minutes)
    log.info(f"[{which}] 重采样 {len(S_clean)} -> {len(Sr)} ({cfg.data.timeslot_minutes}min)")

    # 落盘
    io.save_npz(out_dir / "traffic.npz",
                matrix=g["matrix"].astype(np.float32))
    io.save_pickle({"times": g["times"], "nodes": g["nodes"],
                    "node_coords": g["node_coords"]}, out_dir / "meta.pkl")
    io.save_npz(out_dir / "S_original.npz", S=S.astype(np.float32))
    io.save_npz(out_dir / "S_clean.npz", S_clean=S_clean.astype(np.float32))
    io.save_npz(out_dir / "S_resampled.npz", S_resampled=Sr.astype(np.float32))
    io.save_pickle({"times": tr}, out_dir / "resampled_times.pkl")
    io.save_npz(out_dir / "p_od.npz", p_od=p_od.astype(np.float32))
    with open(out_dir / "clean_stats.json", "w", encoding="utf-8") as f:
        json.dump(clip_stats, f, ensure_ascii=False, indent=2)

    # 诊断图
    diag.plot_s_timeseries(S_clean, g["times"], which.upper(),
                           diag_dir / f"{which}_S_timeseries.png")
    diag.plot_diurnal_boxplot(S_clean, g["times"], g["granularity_min"], which.upper(),
                              diag_dir / f"{which}_diurnal_boxplot.png")
    # 自相关：取约 3 天的 lag
    lags = min(len(S_clean) - 1, int(3 * 24 * 60 / g["granularity_min"]))
    diag.plot_autocorr(S_clean, lags, which.upper(),
                       diag_dir / f"{which}_autocorr.png")
    diag.plot_resample_spectrum(S_clean, g["granularity_min"], Sr,
                                cfg.data.timeslot_minutes, which.upper(),
                                diag_dir / f"{which}_resample_spectrum.png")

    # 原始文件 SHA256
    orig_file = PROJECT_ROOT / cfg[which].file
    sha = _sha256(orig_file)

    meta = {
        "network": which,
        "source_file": cfg[which].file,
        "sha256": sha,
        "n_matrices_actual": int(g["matrix"].shape[0]),
        "n_matrices_nominal": int(cfg[which].n_matrices_expected),
        "n_nodes": n_nodes,
        "n_demands_max": n_nodes * (n_nodes - 1),
        "granularity_min": g["granularity_min"],
        "unit": g["unit"],
        "time_start": g["times"][0].isoformat(),
        "time_end": g["times"][-1].isoformat(),
        "n_time_gaps": len(gaps),
        "resampled_to_min": int(cfg.data.timeslot_minutes),
        "resampled_len": len(Sr),
        "clean_stats": clip_stats,
        "interim_files": [
            f"{which}/traffic.npz", f"{which}/meta.pkl",
            f"{which}/S_original.npz", f"{which}/S_clean.npz",
            f"{which}/S_resampled.npz", f"{which}/resampled_times.pkl",
            f"{which}/p_od.npz", f"{which}/clean_stats.json",
        ],
    }
    # 记录指标
    rec.log_metric(f"{which}_n_matrices", int(g["matrix"].shape[0]))
    rec.log_metric(f"{which}_n_nodes", n_nodes)
    rec.log_metric(f"{which}_n_time_gaps", len(gaps))
    rec.log_metric(f"{which}_S_mean_clean", float(S_clean.mean()))
    rec.log_metric(f"{which}_S_median_clean", float(clip_stats["clip_median"]))
    rec.log_metric(f"{which}_resampled_len", len(Sr))
    return meta


def _process_worldpop(cfg, paths, log, rec) -> dict:
    out_dir = paths["data_interim"] / "worldpop"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("[worldpop] 加载栅格...")
    wp = load_worldpop(cfg)
    log.info(f"[worldpop] {wp['width']}x{wp['height']} crs={wp['crs']}")
    r = compute_region_weights(wp, cfg)
    log.info(f"[worldpop] K={len(r['weights'])} total_pop={r['total_pop']:.0f} "
             f"weights_sum={float(r['weights'].sum()):.6f}")

    io.save_npz(out_dir / "weights.npz",
                valid_rows=r["valid_rows"], valid_cols=r["valid_cols"],
                weights=r["weights"], lon=r["lon"], lat=r["lat"],
                utc_offset=r["utc_offset"])
    io.save_pickle({"shape": r["shape"], "transform": r["transform"],
                    "total_pop": r["total_pop"], "crs": wp["crs"]},
                   out_dir / "meta.pkl")

    orig_file = PROJECT_ROOT / cfg.worldpop.file
    sha = _sha256(orig_file)
    meta = {
        "source_file": cfg.worldpop.file,
        "sha256": sha,
        "width": wp["width"], "height": wp["height"],
        "crs": wp["crs"], "nodata": float(wp["nodata"]),
        "resolution_deg": float(cfg.worldpop.resolution_deg),
        "n_valid_pixels": int(len(r["weights"])),
        "total_pop": float(r["total_pop"]),
        "weights_sum": float(r["weights"].sum()),
        "utc_offset_min": float(r["utc_offset"].min()),
        "utc_offset_max": float(r["utc_offset"].max()),
        "interim_files": ["worldpop/weights.npz", "worldpop/meta.pkl"],
    }
    rec.log_metric("worldpop_K_valid", int(len(r["weights"])))
    rec.log_metric("worldpop_total_pop", float(r["total_pop"]))
    rec.log_metric("worldpop_weights_sum", float(r["weights"].sum()))
    return meta


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p1_main", "P1")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P1")
    paths = resolve_paths(cfg)
    paths["data_interim"].mkdir(parents=True, exist_ok=True)

    log.info("=== P1 数据接入与预处理 ===")

    geant_meta = _process_traffic(cfg, "geant", paths, log, rec)
    abilene_meta = _process_traffic(cfg, "abilene", paths, log, rec)
    wp_meta = _process_worldpop(cfg, paths, log, rec)

    # 数据字典
    data_dict = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "geant": geant_meta,
        "abilene": abilene_meta,
        "worldpop": wp_meta,
    }
    import yaml
    with open(paths["data_interim"] / "DATA_DICT.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_dict, f, allow_unicode=True, sort_keys=False)
    log.info(f"数据字典写入 {paths['data_interim'] / 'DATA_DICT.yaml'}")

    rec.log_output(str(paths["data_interim"]))
    out = rec.finish(status="success")
    log.info(f"P1 完成，记录落盘: {out}")
    log.info("P1 tracking OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
