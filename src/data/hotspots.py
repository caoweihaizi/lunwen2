"""热点 OD 对自动选择。

按 WorldPop 高人口区域选源/目的桶，覆盖主要人口走廊。
"""
from __future__ import annotations

import numpy as np

from .demand import _bucketize


def _haversine_km(lon1, lat1, lon2, lat2):
    """大圆距离 km（向量化）。"""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _region_label(lon, lat):
    """粗略给经纬度打区域标签（人工可读）。"""
    if 70 <= lon <= 150 and 10 <= lat <= 55:
        return "东亚"
    if 60 <= lon <= 100 and 5 <= lat <= 35:
        return "南亚"
    if -10 <= lon <= 40 and 35 <= lat <= 60:
        return "欧洲"
    if -130 <= lon <= -60 and 25 <= lat <= 55:
        return "北美东"
    if -130 <= lon <= -100 and 25 <= lat <= 55:
        return "北美西"
    if -50 <= lon <= -20 and -35 <= lat <= 5:
        return "南美"
    if -20 <= lon <= 50 and -35 <= lat <= 0:
        return "非洲"
    if 110 <= lon <= 155 and -40 <= lat <= -10:
        return "大洋洲"
    if 30 <= lon <= 60 and 15 <= lat <= 40:
        return "中东"
    return "其他"


def select_hotspots(wp: dict, K: int = 10):
    """选 K 对地理分散的热点桶对。

    策略：
    1. 每个区域标签选权重最大的代表桶（保证地理分散）；
    2. 跨区域配对，优先大权重组合，距离>3000km；
    3. 不足补同区域远距对。
    返回 [(src_bucket, dst_bucket, label_src, label_dst), ...]
    """
    bucket_id, bw, blon, blat, butc, nb = _bucketize(wp)

    # 每区域代表桶：区域内权重最大的桶
    region_reps = {}  # region -> (bucket, weight)
    for b in range(nb):
        lab = _region_label(blon[b], blat[b])
        if lab == "其他":
            continue
        if lab not in region_reps or bw[b] > region_reps[lab][1]:
            region_reps[lab] = (b, float(bw[b]))

    reps = [(b, w, lab) for lab, (b, w) in region_reps.items()]
    reps.sort(key=lambda x: -x[1])  # 按权重降序

    pairs = []
    used_pairs = set()
    # 跨区域对，优先高权重组合
    for i in range(len(reps)):
        for j in range(len(reps)):
            if i == j:
                continue
            s, _, ls = reps[i]
            d, _, ld = reps[j]
            if ls == ld:
                continue
            dist = _haversine_km(blon[s], blat[s], blon[d], blat[d])
            if dist < 3000:
                continue
            key = tuple(sorted([ls, ld]))
            if key in used_pairs:
                continue
            pairs.append((int(s), int(d), ls, ld))
            used_pairs.add(key)
            if len(pairs) >= K:
                return pairs, (bucket_id, bw, blon, blat, butc, nb)

    # 不足补同区域远距对（用 top 桶内同区域远距）
    top = np.argsort(bw)[::-1][:30]
    for i in range(len(top)):
        for j in range(len(top)):
            if i == j or len(pairs) >= K:
                continue
            s, d = top[i], top[j]
            dist = _haversine_km(blon[s], blat[s], blon[d], blat[d])
            if dist < 2000:
                continue
            if (int(s), int(d)) in [(p[0], p[1]) for p in pairs]:
                continue
            ls = _region_label(blon[s], blat[s])
            ld = _region_label(blon[d], blat[d])
            pairs.append((int(s), int(d), ls, ld))
            if len(pairs) >= K:
                break
    return pairs, (bucket_id, bw, blon, blat, butc, nb)
