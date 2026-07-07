"""基础地面需求（时间模式 × 空间权重 × 昼夜调制）。

按 UTC 偏移分桶，避免逐像元逐时隙计算。桶级需求 = W_b × 昼夜系数(t,b) × S(t) × k0。
"""
from __future__ import annotations

import numpy as np


def _bucketize(wp: dict, bucket_minutes: int = 15):
    """把有效像元按 UTC 偏移分桶。

    返回:
      bucket_ids: (K,) int 每像元的桶 id（0..n_buckets-1）
      bucket_weights: (n_buckets,) 每桶权重和
      bucket_lon, bucket_lat: (n_buckets,) 每桶代表像元经纬度
      bucket_utc: (n_buckets,) 每桶 UTC 偏移（小时）
      n_buckets
    """
    utc = wp["utc_offset"].astype(np.float64)  # 小时
    # 量化到 bucket_minutes 粒度
    bucket_hours = bucket_minutes / 60.0
    bucket_id = np.round(utc / bucket_hours).astype(np.int32)
    bucket_id -= bucket_id.min()
    n_buckets = int(bucket_id.max()) + 1

    w = wp["weights"].astype(np.float64)
    bucket_weights = np.zeros(n_buckets, dtype=np.float64)
    np.add.at(bucket_weights, bucket_id, w)

    # 每桶代表像元：权重最大的那个
    bucket_lon = np.zeros(n_buckets, dtype=np.float64)
    bucket_lat = np.zeros(n_buckets, dtype=np.float64)
    bucket_utc = np.zeros(n_buckets, dtype=np.float64)
    for b in range(n_buckets):
        m = bucket_id == b
        idx = np.where(m)[0]
        # 代表像元取桶内权重最大
        rep = idx[np.argmax(w[idx])]
        bucket_lon[b] = wp["lon"][rep]
        bucket_lat[b] = wp["lat"][rep]
        bucket_utc[b] = utc[rep]

    return (bucket_id, bucket_weights, bucket_lon, bucket_lat, bucket_utc, n_buckets)


def compute_base_demand(S_clean, times, wp: dict, cfg, k0: float = 1.0):
    """桶级基础地面需求时间序列。

    返回:
      {
        "bucket_demand_ts": (T, n_buckets) float64, 每桶每时隙需求(Mbps*任意单位)
        "bucket_weights", "bucket_lon", "bucket_lat", "bucket_utc",
        "bucket_ids": (K,) 每像元的桶 id,
        "n_buckets", "times", "S_clean"
      }
    """
    a = float(cfg.demand.diurnal_amplitude_a)
    bucket_id, bw, blon, blat, butc, nb = _bucketize(wp)

    T = len(times)
    # 每时隙的小时浮点数（用于昼夜调制）
    t_hours = np.array(
        [t.hour + t.minute / 60.0 + t.second / 3600.0 for t in times], dtype=np.float64
    )

    # 当地正午 UTC = 12 - utc_offset；昼夜系数 = 1 + a*cos(2π*(t_utc - local_noon)/24)
    # 但 utc_offset 可能 |x|>12（经度±180），裁剪
    local_noon = 12.0 - butc  # (n_buckets,)
    # 广播：(T,1) - (1,n_buckets)
    phase = 2 * np.pi * (t_hours[:, None] - local_noon[None, :]) / 24.0
    diurnal = 1.0 + a * np.cos(phase)  # (T, n_buckets)
    diurnal = np.clip(diurnal, 0.0, None)  # 防负

    # 桶级需求 = W_b * diurnal(t,b) * S(t) * k0
    bucket_demand_ts = (bw[None, :] * diurnal * S_clean[:, None] * k0)  # (T, n_buckets)

    return {
        "bucket_demand_ts": bucket_demand_ts,
        "bucket_weights": bw,
        "bucket_lon": blon,
        "bucket_lat": blat,
        "bucket_utc": butc,
        "bucket_ids": bucket_id,
        "n_buckets": nb,
        "times": times,
        "S_clean": S_clean,
        "k0": k0,
    }
