"""批量加载 SNDlib 流量矩阵归档 → 统一时间索引的 (T,N,N) 矩阵。

GÉANT/Abilene 的 OD 是稀疏的（GÉANT 441/462，Abilene 131/132），
缺失 OD 对在矩阵中填 0。节点顺序固定为首个文件的 NODES 顺序。
"""
from __future__ import annotations

import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np

from .sndlib_parser import parse_sndlib_native


def _parse_timestamp(ts: str) -> datetime:
    """'20050504-1630' → datetime(2005,5,4,16,30)。"""
    ts = ts.strip()
    # YYYYMMDD-HHMM
    return datetime.strptime(ts, "%Y%m%d-%H%M")


def load_traffic_matrix_archive(cfg, which: str) -> dict:
    """加载 cfg.<which>.file 指向的 tgz，返回统一矩阵。

    Args:
      cfg: OmegaConf 配置。
      which: 'geant' 或 'abilene'。

    Returns:
      {
        "network", "times": list[datetime], "nodes": list[str],
        "node_coords": list[(lon,lat)], "matrix": np.ndarray (T,N,N) float64 Mbps,
        "granularity_min": int, "unit": str,
      }
    """
    section = cfg[which]
    fname = section.file
    n_expected = int(section.n_matrices_expected)

    # 定位 tgz：项目根下
    from src.common.config import PROJECT_ROOT

    tgz_path = PROJECT_ROOT / fname
    if not tgz_path.exists():
        raise FileNotFoundError(f"流量矩阵归档不存在: {tgz_path}")

    # 第一遍：抽取所有文件文本，解析
    parsed = []  # [(datetime, result_dict)]
    node_order = None
    node_coords = None
    unit = ""
    granularity_min = 0

    with tarfile.open(tgz_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".txt")]
        for m in members:
            text = tar.extractfile(m).read().decode("utf-8", errors="replace")
            r = parse_sndlib_native(text)
            if not r["nodes"] or not r["demands"]:
                continue
            # 固定节点顺序为首个有效文件
            if node_order is None:
                node_order = [nid for nid, _, _ in r["nodes"]]
                node_coords = [(lon, lat) for _, lon, lat in r["nodes"]]
                unit = r["unit"]
                gran = r["granularity"]
                granularity_min = 15 if "15" in gran else (5 if "5" in gran else 0)
            # 校验节点集一致
            cur_nodes = {nid for nid, _, _ in r["nodes"]}
            if cur_nodes != set(node_order):
                # 节点集变化，跳过并记录（不应发生）
                continue
            try:
                dt = _parse_timestamp(r["timestamp"])
            except ValueError:
                continue
            parsed.append((dt, r))

    if not parsed:
        raise RuntimeError(f"{which}: 未解析到任何有效矩阵")

    # 按时间排序
    parsed.sort(key=lambda x: x[0])
    times = [dt for dt, _ in parsed]

    n = len(node_order)
    idx = {nid: i for i, nid in enumerate(node_order)}
    T = len(parsed)
    matrix = np.zeros((T, n, n), dtype=np.float64)

    for t, (_, r) in enumerate(parsed):
        for src, dst, val in r["demands"]:
            i, j = idx[src], idx[dst]
            matrix[t, i, j] = val

    # 矩阵数核对
    if T != n_expected:
        # 不阻断，但调用方应感知
        import warnings
        warnings.warn(f"{which}: 矩阵数 {T} != 预期 {n_expected}")

    return {
        "network": which,
        "times": times,
        "nodes": node_order,
        "node_coords": node_coords,
        "matrix": matrix,
        "granularity_min": granularity_min,
        "unit": unit,
    }


def detect_time_gaps(times, granularity_min: int) -> list:
    """检测时间缺口，返回 [(t_i, t_{i+1}, gap_minutes)] 列表。"""
    gaps = []
    expected = granularity_min
    for a, b in zip(times, times[1:]):
        delta = (b - a).total_seconds() / 60.0
        if abs(delta - expected) > 1e-6:
            gaps.append((a, b, delta))
    return gaps
