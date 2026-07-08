"""链路故障注入（技术大纲 §4.5.2）。

10% 链路故障，固定故障集（全程），故障先于路由生效。
"""
from __future__ import annotations

import numpy as np


def inject_failures(edge_lists, cfg, rng, scenario="compound"):
    """生成固定故障边集。

    scenario:
      'baseline' — 无故障
      'compound' — 10% 链路故障（固定集，全程失效）

    返回 set of frozenset({i,j}) 故障边。
    """
    if scenario == "baseline":
        return set()

    rate = float(cfg.failure.link_failure_rate)
    # 从首个时隙拓扑取所有无向边
    edges0 = edge_lists[0]
    n_edges = len(edges0)
    n_fail = max(1, int(round(n_edges * rate)))
    # 随机选 n_fail 条边
    chosen = rng.choice(n_edges, size=n_fail, replace=False)
    failed = set()
    for k in chosen:
        i, j = int(edges0[k, 0]), int(edges0[k, 1])
        failed.add(frozenset({i, j}))
    return failed
