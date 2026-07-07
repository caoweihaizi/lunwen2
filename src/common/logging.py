"""统一日志：同时输出到控制台与文件。

get_logger(name, stage) 返回带 stage 前缀的 logger。
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import PROJECT_ROOT

_LOG_DIR = PROJECT_ROOT / "logs"
_CONFIGURED = set()  # 已配置的 (stage) 记录，避免重复 addHandler


def get_logger(name: str, stage: str) -> logging.Logger:
    """返回带 stage 前缀的 logger，同时输出控制台与 logs/<stage>_*.log。"""
    logger = logging.getLogger(f"{stage}:{name}")
    if stage in _CONFIGURED:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(_LOG_DIR / f"{stage}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _CONFIGURED.add(stage)
    return logger
