"""遥测缺失注入（技术大纲 §4.4）。

三类机制，均在 link_state 真值副本上操作（不改真值）。
输入 shard: (M,10) 列 [t,i,j,offered,carried,queue,drop,util,delay,dist]
列索引: offered=3, carried=4, queue=5, drop=6, util=7, delay=8

返回 (mask, observed, stats):
  mask: (M, n_fields) bool, True=观测到, False=缺失
  observed: (M, 10) 副本，缺失字段置 NaN
  stats: 缺失统计
"""
from __future__ import annotations

import numpy as np

# 默认注入缺失的字段（流量/利用率/队列）
FIELD_COLS = (4, 5, 7)  # carried, queue, util


def inject_mcar(ls, rate, rng, field_cols=FIELD_COLS):
    """MCAR：每条记录每字段独立以 rate 概率缺失。"""
    M = len(ls)
    n_f = len(field_cols)
    mask = rng.random((M, n_f)) >= rate  # True=观测
    observed = ls.astype(np.float64).copy()
    for k, col in enumerate(field_cols):
        observed[~mask[:, k], col] = np.nan
    actual = 1.0 - mask.mean()
    stats = {"mechanism": "mcar", "target_rate": rate, "actual_rate": float(actual),
             "n_fields": n_f}
    return mask, observed, stats


def inject_mar(ls, target_rate, rng, field_cols=FIELD_COLS, a=2.0, b=1.0):
    """MAR：缺失概率随 util/queue 增大。σ(aU + bQ̃ + c)，标定 c 使总缺失率≈target。

    U=util(列7), Q̃=queue/buffer(列5归一)。
    返回 (mask, observed, stats)。
    """
    M = len(ls)
    n_f = len(field_cols)
    util = ls[:, 7].astype(np.float64)
    # queue 归一化：buffer = capacity（QueueState 里 buffer=capacity×1时隙）
    # 但 shard 里没有 capacity 列，用 queue 的 max 近似归一
    q = ls[:, 5].astype(np.float64)
    q_max = max(q.max(), 1e-9)
    q_norm = q / q_max

    # 二分 c 使总缺失率达 target
    def miss_rate(c):
        logits = a * util + b * q_norm + c
        p_miss = 1.0 / (1.0 + np.exp(-logits))  # σ
        return p_miss.mean()

    lo, hi = -10.0, 10.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if miss_rate(mid) < target_rate:
            lo = mid
        else:
            hi = mid
    c = (lo + hi) / 2
    logits = a * util + b * q_norm + c
    p_miss = 1.0 / (1.0 + np.exp(-logits))

    # 每条记录每字段独立按 p_miss 采样
    mask = rng.random((M, n_f)) >= p_miss[:, None]
    observed = ls.astype(np.float64).copy()
    for k, col in enumerate(field_cols):
        observed[~mask[:, k], col] = np.nan

    # 高负载 vs 低负载缺失率（验证 MAR 非退化）
    high = util >= 0.8
    low = util < 0.2
    p_high = p_miss[high].mean() if high.any() else 0.0
    p_low = p_miss[low].mean() if low.any() else 0.0
    stats = {"mechanism": "mar", "target_rate": target_rate,
             "actual_rate": float(1 - mask.mean()), "a": a, "b": b, "c": float(c),
             "high_load_miss": float(p_high), "low_load_miss": float(p_low),
             "n_high": int(high.sum()), "n_low": int(low.sum())}
    return mask, observed, stats


def inject_block(ls, block_len, rng, field_cols=FIELD_COLS, target_rate=0.20):
    """Block：随机选链路，连续 block_len 时隙缺失，选链路数使总缺失率≈target。

    shard 内：按 (i,j) 分组，每组若被选中则连续 block_len 时隙缺失。
    """
    M = len(ls)
    n_f = len(field_cols)
    mask = np.ones((M, n_f), dtype=bool)
    # 按 (i,j) 分组
    edges = {}
    for idx in range(M):
        key = (int(ls[idx, 1]), int(ls[idx, 2]))
        edges.setdefault(key, []).append(idx)
    # 估算需选多少链路：总记录 M, 每条链路平均 M/n_edges 条，block_len 缺失
    n_edges = len(edges)
    # 每选一条链路缺失 block_len 条记录（×n_fields）
    # 总缺失率 ≈ (n_selected × block_len × n_fields) / (M × n_fields) = n_selected×block_len/M
    n_select = max(1, int(round(target_rate * M / block_len)))
    n_select = min(n_select, n_edges)
    edge_keys = list(edges.keys())
    sel_idx = rng.choice(len(edge_keys), size=n_select, replace=False)
    selected = [edge_keys[i] for i in sel_idx]
    for ek in selected:
        idxs = edges[ek]
        # 在该链路的记录里随机选一个起点，连续 block_len 条缺失
        if len(idxs) <= block_len:
            start = 0
            span = idxs
        else:
            start = rng.randint(0, len(idxs) - block_len)
            span = idxs[start:start + block_len]
        for idx in span:
            mask[idx, :] = False
    observed = ls.astype(np.float64).copy()
    for k, col in enumerate(field_cols):
        observed[~mask[:, k], col] = np.nan
    actual = 1.0 - mask.mean()
    stats = {"mechanism": "block", "target_rate": target_rate, "block_len": block_len,
             "actual_rate": float(actual), "n_selected_links": int(n_select),
             "n_edges": int(n_edges)}
    return mask, observed, stats


def compute_age(mask, n_timesteps):
    """信息年龄：每条记录每字段，距离上次观测（mask=True）的时隙数。

    mask: (M, n_f) 按时隙有序。需按 (i,j) 分组内按 t 累计。
    这里简化：输入的 ls 也一并传入以分组。
    返回 age: (M, n_f) int，从未观测过的为累计时隙数。
    """
    pass  # age 在 observed.py 里按 shard 顺序算（需跨 shard，P5 main 处理）
