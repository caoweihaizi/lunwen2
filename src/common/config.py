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


def load_config(overrides: dict | None = None, cli: bool = True):
    """加载全局配置，可选 overlay override。

    - overrides: 显式 dict，优先级高于文件、低于命令行。
    - cli: 是否解析 sys.argv 中的 key=value 形式参数（OmegaConf.from_cli），
      优先级最高。run.sh 通过 `python -m src.pN_main seed.data=3 ...` 传入。
    """
    cfg = OmegaConf.load(CONFIG_PATH)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    if cli:
        # OmegaConf.from_cli 解析 sys.argv[1:] 中的 a=b 形式参数
        cli_cfg = OmegaConf.from_cli()
        if cli_cfg:
            cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def config_hash(cfg) -> str:
    """对配置做稳定序列化（排序键）后取 SHA256 前 12 位。

    seed 字段被排除——它是运行实例标识而非配置内容：同一配置跑 5 个种子
    应共享同一 config_hash，便于 P15 按配置归组。
    待回填的占位参数（walker、k0）随阶段推进会变化，属于预期行为，
    跨阶段对账用 (stage, code_version, seed) 而非纯 config_hash。
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    # 剥离 seed：它标识运行实例，不标识配置
    if isinstance(resolved, dict) and "seed" in resolved:
        resolved = {k: v for k, v in resolved.items() if k != "seed"}
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
