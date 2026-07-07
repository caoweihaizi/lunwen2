"""时间顺序划分（训练/验证/校准/测试）。

禁止跨段滑动窗口——本模块只产出分段索引，窗口生成在 P4/P7 各段内部。
"""
from __future__ import annotations


def time_split(times, ratios=(0.60, 0.15, 0.10, 0.15)):
    """按时间顺序切四段。

    times: 长度 T 的时刻列表（已按时间递增）。
    ratios: (train, val, calib, test)。
    返回 {'train':(s,e), 'val':(s,e), 'calib':(s,e), 'test':(s,e)} 闭开区间。
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    T = len(times)
    names = ["train", "val", "calib", "test"]
    cum = 0
    out = {}
    for i, name in enumerate(names):
        start = int(round(cum * T))
        cum += ratios[i]
        end = int(round(cum * T)) if i < len(names) - 1 else T
        out[name] = (start, end)
    # 防御：保证不重叠且覆盖全段
    return out
