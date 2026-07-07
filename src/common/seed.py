"""全局种子管理。

提供：
- seed_everything: 同时设置 random/numpy/torch（CPU 与 MPS）种子。
- make_seed_stream: 基于 (base, kind) 生成可复现的子种子流。
"""
from __future__ import annotations

import hashlib
import random
from typing import Iterator

import numpy as np


def seed_everything(seed: int) -> None:
    """设置 random / numpy / torch 的全局种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def make_seed_stream(base: int, kind: str) -> Iterator[int]:
    """基于 (base, kind) 生成可复现的子种子流。

    同一 (base, kind) 多次调用产出相同序列；不同 kind 产出不同序列。
    供同一阶段内多个随机组件独立但不冲突地取种子。
    """
    # 用 numpy RandomState，避免扰动全局 np.random 状态。
    # 用 hashlib 保证跨进程稳定（避免 PYTHONHASHSEED 导致 hash() 不稳定）。
    digest = hashlib.sha256(f"{base}|{kind}".encode("utf-8")).digest()
    h = int.from_bytes(digest[:4], "little") & 0xFFFFFFFF
    rng = np.random.RandomState(h)
    while True:
        yield int(rng.randint(0, 2**31 - 1))
