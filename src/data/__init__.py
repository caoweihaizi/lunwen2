"""P1 数据接入与预处理。"""
from .load_traffic import load_traffic_matrix_archive, detect_time_gaps
from .sndlib_parser import parse_sndlib_native
from .time_patterns import (
    extract_global_intensity,
    extract_od_relative_pattern,
    winsorize_intensity,
)
from .resample import resample_to_timeslot
from .worldpop import load_worldpop, compute_region_weights
from . import io

__all__ = [
    "load_traffic_matrix_archive",
    "detect_time_gaps",
    "parse_sndlib_native",
    "extract_global_intensity",
    "extract_od_relative_pattern",
    "winsorize_intensity",
    "resample_to_timeslot",
    "load_worldpop",
    "compute_region_weights",
    "io",
]
