"""WorldPop 栅格读取与区域需求权重。

稠密 (H,W) 栅格 → 稀疏有效像元三元组（全局工程约定：禁存稠密 0 填充）。
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import xy

from src.common.config import PROJECT_ROOT


def load_worldpop(cfg) -> dict:
    """打开 cfg.worldpop.file，返回栅格元数据 + masked 数据。"""
    path = PROJECT_ROOT / cfg.worldpop.file
    if not path.exists():
        raise FileNotFoundError(f"WorldPop 栅格不存在: {path}")
    with rasterio.open(path) as ds:
        data = ds.read(1, masked=True)  # masked array, nodata 自动掩
        return {
            "data": data,
            "transform": ds.transform,
            "crs": str(ds.crs),
            "bounds": ds.bounds,
            "width": ds.width,
            "height": ds.height,
            "nodata": ds.nodata,
        }


def compute_region_weights(worldpop: dict, cfg) -> dict:
    """计算有效像元的归一化权重与 UTC 偏移（稀疏三元组）。

    返回:
      {
        "valid_rows","valid_cols","weights","lon","lat","utc_offset",
        "total_pop","shape","transform"
      }
    """
    data = worldpop["data"]
    H, W = worldpop["height"], worldpop["width"]
    transform = worldpop["transform"]

    # 有效像元：非 nodata、非 mask、pop>0
    arr = np.asarray(data, dtype=np.float64)
    mask = np.ma.getmaskarray(data) if np.ma.isMaskedArray(data) else np.zeros_like(arr, dtype=bool)
    valid = (~mask) & (arr > 0)

    rows, cols = np.nonzero(valid)
    pop = arr[rows, cols]
    total_pop = float(pop.sum())
    weights = (pop / total_pop).astype(np.float32) if total_pop > 0 else np.zeros(len(pop), dtype=np.float32)

    # 每个有效像元的中心经纬度
    lons = np.empty(len(rows), dtype=np.float32)
    lats = np.empty(len(rows), dtype=np.float32)
    # xy 支持数组输入
    lons[:], lats[:] = xy(transform, rows, cols, offset="center")
    utc_offset = (lons / 15.0).astype(np.float32)

    return {
        "valid_rows": rows.astype(np.int32),
        "valid_cols": cols.astype(np.int32),
        "weights": weights,
        "lon": lons,
        "lat": lats,
        "utc_offset": utc_offset,
        "total_pop": total_pop,
        "shape": (H, W),
        "transform": transform[:6] if hasattr(transform, "__len__") else transform,
    }
