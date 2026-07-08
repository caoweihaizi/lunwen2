"""观测表生成：真值 + 缺失掩码 + 观测值 + 信息年龄（技术大纲 §4.3.4）。

shard 内按 (i,j) 分组按 t 累计 age。跨 shard 的 age 在 p5_main 用上一 shard 末状态接续。
"""
from __future__ import annotations

import numpy as np

from .inject import FIELD_COLS


def add_age(observed, mask, ls, field_cols=FIELD_COLS):
    """在 shard 内按 (i,j) 分组累计 age。

    observed: (M,10) 副本（缺失处 NaN）
    mask: (M, n_f) True=观测
    ls: (M,10) 原始真值（用于取 t,i,j）
    返回 age: (M, n_f) int，每条记录每字段的年龄。
    """
    M = len(ls)
    n_f = len(field_cols)
    age = np.zeros((M, n_f), dtype=np.int32)
    # 按 (i,j) 分组，组内按 t 排序（shard 内已按 t 顺序，但同一 t 多链路）
    edges = {}
    for idx in range(M):
        key = (int(ls[idx, 1]), int(ls[idx, 2]))
        edges.setdefault(key, []).append(idx)
    for key, idxs in edges.items():
        last_observed = [-1] * n_f  # 上次观测到的时隙号
        # shard 起始年龄由 p5_main 传入的 prev_age_map 提供（跨 shard 接续）
        # 这里只算 shard 内，prev_age 通过 age_init 参数传入
        for idx in idxs:
            t = int(ls[idx, 2]) if False else int(ls[idx, 0])  # t 是列0
            for k in range(n_f):
                if mask[idx, k]:
                    age[idx, k] = 0
                    last_observed[k] = t
                else:
                    # 年龄 = 距上次观测的时隙数；若从未观测，用累计
                    if last_observed[k] >= 0:
                        age[idx, k] = t - last_observed[k]
                    else:
                        age[idx, k] = t + 1  # 从未观测，按 t+1 计（shard 内）
    return age


def build_observed_shard(ls, mechanism, cfg, rng, prev_age_state=None):
    """对一个 shard 生成完整观测表。

    返回 dict: ls_truth, mask, observed, age, stats, field_cols, plus 更新的 prev_age_state。
    prev_age_state: {(i,j): [last_t_per_field]} 跨 shard 接续，可为 None。
    """
    from .inject import inject_mcar, inject_mar, inject_block
    n_f = len(FIELD_COLS)
    if mechanism == "mcar20":
        mask, obs, stats = inject_mcar(ls, 0.20, rng)
    elif mechanism == "mcar40":
        mask, obs, stats = inject_mcar(ls, 0.40, rng)
    elif mechanism == "mar20":
        mask, obs, stats = inject_mar(ls, 0.20, rng)
    elif mechanism == "block10":
        mask, obs, stats = inject_block(ls, int(cfg.missing.block_length), rng,
                                        target_rate=0.20)
    else:
        raise ValueError(f"未知机制 {mechanism}")

    # age（shard 内 + 跨 shard 接续）
    age = np.zeros((len(ls), n_f), dtype=np.int32)
    edges = {}
    for idx in range(len(ls)):
        key = (int(ls[idx, 1]), int(ls[idx, 2]))
        edges.setdefault(key, []).append(idx)
    state = prev_age_state or {}
    for key, idxs in edges.items():
        last_t = state.get(key, [-1] * n_f)
        for idx in idxs:
            t = int(ls[idx, 0])
            for k in range(n_f):
                if mask[idx, k]:
                    age[idx, k] = 0
                    last_t[k] = t
                else:
                    age[idx, k] = (t - last_t[k]) if last_t[k] >= 0 else (t + 1)
        state[key] = last_t

    return {
        "ls_truth": ls, "mask": mask, "observed": obs, "age": age,
        "stats": stats, "field_cols": list(FIELD_COLS), "prev_age_state": state,
    }
