"""种子管理单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.seed import make_seed_stream, seed_everything


def take(stream, n):
    return [next(stream) for _ in range(n)]


def test_seed_stream_reproducible_same_base_kind():
    a = take(make_seed_stream(0, "data"), 5)
    b = take(make_seed_stream(0, "data"), 5)
    assert a == b, f"same (base,kind) must be reproducible: {a} vs {b}"


def test_seed_stream_diff_kind():
    a = take(make_seed_stream(0, "data"), 5)
    b = take(make_seed_stream(0, "model"), 5)
    assert a != b, "different kind must produce different sequences"


def test_seed_stream_diff_base():
    a = take(make_seed_stream(0, "data"), 5)
    b = take(make_seed_stream(1, "data"), 5)
    assert a != b, "different base must produce different sequences"


def test_seed_everything_runs():
    # 不报错即可；numpy 全局状态可复现。
    seed_everything(42)
    import numpy as np
    x = np.random.rand(3)
    seed_everything(42)
    y = np.random.rand(3)
    assert np.allclose(x, y), "seed_everything must make numpy reproducible"
