"""内部格式读写。

大数组用 .npz（压缩），元数据用 .pkl。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def save_npz(path: Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    # .npz 后缀由 np 自动加
    return Path(str(path) + ("" if str(path).endswith(".npz") else ".npz"))


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_pickle(obj, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
