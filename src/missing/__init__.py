"""P5 遥测缺失观测生成。"""
from .inject import inject_mcar, inject_mar, inject_block, FIELD_COLS
from .observed import build_observed_shard, add_age

__all__ = [
    "inject_mcar", "inject_mar", "inject_block", "FIELD_COLS",
    "build_observed_shard", "add_age",
]
