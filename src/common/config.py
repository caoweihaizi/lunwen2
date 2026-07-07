"""配置加载与哈希。

提供：
- load_config: 加载 configs/config.yaml，支持命令行 override。
- config_hash: 对配置做稳定序列化后取 SHA256 前 12 位。
- resolve_paths: 返回各目录的绝对路径。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

# 项目根目录：本文件位于 src/common/config.py，上溯两级。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(overrides: dict | None = None):
    """加载全局配置，可选 overlay override。返回 OmegaConf 对象。"""
    cfg = OmegaConf.load(CONFIG_PATH)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def config_hash(cfg) -> str:
    """对配置做稳定序列化（排序键）后取 SHA256 前 12 位。

    OmegaConf 解析出的容器经 json 序列化时键已排序，保证跨运行稳定。
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    blob = json.dumps(resolved, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def resolve_paths(cfg) -> dict:
    """返回关键目录的绝对路径，保证跨机器可移植。"""
    root = PROJECT_ROOT
    return {
        "root": root,
        "data_raw": root / "data" / "raw",
        "data_interim": root / "data" / "interim",
        "data_processed": root / "data" / "processed",
        "models": root / "models",
        "results_metrics": root / "results" / "metrics",
        "logs": root / "logs",
    }
