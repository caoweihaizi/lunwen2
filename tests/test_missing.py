"""缺失注入单元测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.missing import inject_mcar, inject_mar, inject_block


def _sample_ls(n=1000):
    """造 (n,10) 真值：t,i,j,offered,carried,queue,drop,util,delay,dist。"""
    rng = np.random.RandomState(0)
    t = np.arange(n) % 100  # 100 时隙
    i = (np.arange(n) % 5).astype(float)
    j = ((np.arange(n) + 1) % 5).astype(float)
    util = rng.random(n)  # 0-1
    carried = util * 1000
    queue = rng.random(n) * 500
    offered = carried + queue
    drop = np.zeros(n)
    delay = rng.random(n) * 15
    dist = rng.random(n) * 4000
    return np.stack([t, i, j, offered, carried, queue, drop, util, delay, dist], axis=1)


def test_mcar_rate():
    rng = np.random.RandomState(1)
    ls = _sample_ls()
    mask, obs, stats = inject_mcar(ls, 0.20, rng)
    actual = 1 - mask.mean()
    assert abs(actual - 0.20) < 0.02, f"MCAR 实际缺失率 {actual:.3f} 偏离 0.20"
    # 缺失处为 NaN（每字段各自的 mask）
    for k, col in enumerate((4, 5, 7)):
        assert np.isnan(obs[~mask[:, k], col]).all()


def test_mcar_truth_unchanged():
    rng = np.random.RandomState(2)
    ls = _sample_ls()
    ls_copy = ls.copy()
    inject_mcar(ls, 0.30, rng)
    assert np.array_equal(ls, ls_copy), "真值不应被修改"


def test_mar_high_vs_low_load():
    rng = np.random.RandomState(3)
    ls = _sample_ls()
    mask, obs, stats = inject_mar(ls, 0.20, rng)
    actual = 1 - mask.mean()
    assert abs(actual - 0.20) < 0.03, f"MAR 实际缺失率 {actual:.3f} 偏离 0.20"
    # 高负载链路缺失率应显著高于低负载
    assert stats["high_load_miss"] > stats["low_load_miss"], \
        f"MAR 退化：高负载{stats['high_load_miss']:.3f} <= 低负载{stats['low_load_miss']:.3f}"


def test_block_continuous():
    rng = np.random.RandomState(4)
    ls = _sample_ls(2000)
    mask, obs, stats = inject_block(ls, 10, rng, target_rate=0.20)
    # 检查存在连续缺失：某链路连续 10 条 mask=False
    # 至少有一组连续缺失长度 >= 5
    found = False
    edges = {}
    for idx in range(len(ls)):
        key = (int(ls[idx, 1]), int(ls[idx, 2]))
        edges.setdefault(key, []).append(idx)
    for key, idxs in edges.items():
        miss = ~mask[idxs, 0]
        # 找最长连续 False
        max_run = 0; cur = 0
        for m in miss:
            if not m:
                cur += 1; max_run = max(max_run, cur)
            else:
                cur = 0
        if max_run >= 5:
            found = True
            break
    assert found, "Block 缺失未产生连续段"
