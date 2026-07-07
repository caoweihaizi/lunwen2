"""P0 阶段自检入口。

无真正实验，只做工程链路自检：
- 加载配置并打印哈希
- 设置种子
- 记录一次 ExperimentRecord（含 dummy 指标）
- 打印目录树与依赖版本
落盘 results/metrics/<hash>_0_P0.json 作为 P0 完成证据。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 保证可 `python -m src.p0_main` 运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    ExperimentRecord,
    config_hash,
    get_logger,
    load_config,
    resolve_paths,
    seed_everything,
)


def _dir_tree(root: Path, max_depth: int = 2) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        if any(part in {".venv", ".harness", "__pycache__"} for part in p.parts):
            continue
        rel = p.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        lines.append(("  " * (depth - 1)) + ("- " if p.is_dir() else "  ") + rel.name)
    return "\n".join(lines)


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p0_main", "P0")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P0")

    chash = config_hash(cfg)
    log.info(f"P0 自检启动 | config_hash={chash}")

    # 依赖版本
    import numpy, torch
    try:
        import rasterio
        rasterio_v = rasterio.__version__
    except ImportError:
        rasterio_v = "MISSING"
    log.info(f"python={sys.version.split()[0]} numpy={numpy.__version__} "
             f"torch={torch.__version__} rasterio={rasterio_v} "
             f"mps={torch.backends.mps.is_available()}")

    # 目录树
    paths = resolve_paths(cfg)
    log.info(f"项目根: {paths['root']}")
    log.info("目录树:\n" + _dir_tree(paths["root"], max_depth=2))

    # dummy 指标，证明 metrics 链路通
    rec.log_metric("selfcheck", 1.0)
    rec.log_metric("torch_mps_available", bool(torch.backends.mps.is_available()))

    out = rec.finish(status="success")
    log.info(f"P0 完成，记录落盘: {out}")
    log.info("P0 tracking OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
