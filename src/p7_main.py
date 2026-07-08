"""P7 阶段入口：预测模型实现与训练。

多策略混合预训练（9 run × mcar20），冻结预测器供 P8/P10。
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    ExperimentRecord, get_logger, load_config, resolve_paths, seed_everything,
)
from src.predict.dataset import LinkStateDataset  # noqa: E402
from src.predict.train import train_model  # noqa: E402


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p7_main", "P7")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P7")
    paths = resolve_paths(cfg)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"=== P7 预测模型训练 (device={device}) ===")

    # 9 run × mcar20（多策略混合）
    runs = []
    for pol in ["dijkstra", "ecmp", "queue_aware"]:
        for scn in ["baseline", "burst", "compound"]:
            runs.append((f"{pol}_{scn}", "mcar20"))
    log.info(f"训练数据: {len(runs)} run × mcar20")

    # 划分
    split = pickle.load(open(paths["data_processed"] / "demand" / "split.pkl", "rb"))["split"]

    # 训练集（train 段，2000 窗/run）
    log.info("构建训练集...")
    t0 = time.time()
    train_ds = LinkStateDataset(cfg, runs, split["train"], n_windows=2000,
                                W=int(cfg.data.history_window_W),
                                H=tuple(cfg.data.forecast_horizons),
                                seed=cfg.seed.data)
    log.info(f"训练集: {len(train_ds)} 样本, {time.time()-t0:.0f}s, "
             f"normalize mean={train_ds.normalize['mean']}")

    # 验证集（val 段，500 窗/run，复用训练集 normalize）
    val_ds = LinkStateDataset(cfg, runs, split["val"], n_windows=500,
                              W=int(cfg.data.history_window_W),
                              H=tuple(cfg.data.forecast_horizons),
                              seed=cfg.seed.data + 1,
                              normalize=train_ds.normalize)
    log.info(f"验证集: {len(val_ds)} 样本")

    # 训练（调参版：hidden=128, 2层GRU, 50epoch, LR调度, 梯度裁剪）
    log.info("开始训练（调参版 hidden=128 2层GRU 50epoch LR调度）...")
    model, stats = train_model(cfg, train_ds, val_ds, log, epochs=50,
                               batch_size=256, lr=1e-3, device=device,
                               hidden=128, n_gru_layers=2)

    # 落盘 checkpoint
    ckpt_dir = paths["models"] / "predict"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "predict_v1.pt"
    torch.save({"state_dict": model.state_dict(),
                "normalize": train_ds.normalize,
                "config": {"n_feat": 12, "hidden": 128, "n_h": 2, "h_max": 3,
                           "n_gru_layers": 2,
                           "W": int(cfg.data.history_window_W),
                           "H": list(cfg.data.forecast_horizons)}},
               ckpt_path)
    log.info(f"checkpoint 落盘: {ckpt_path}")

    # 推理时间
    model.eval()
    x, future, y = train_ds[0]
    x = x.unsqueeze(0).to(device); future = future.unsqueeze(0).to(device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x, future)
    if device == "mps":
        torch.mps.synchronize()
    infer_ms = (time.time() - t0) / 100 * 1000
    n_params = sum(p.numel() for p in model.parameters())

    # 指标
    rec.log_metric("best_val_loss", stats["best_val_loss"])
    rec.log_metric("crossing_rate", stats["final_crossing_rate"])
    rec.log_metric("n_params", n_params)
    rec.log_metric("infer_ms", infer_ms)
    rec.log_metric("n_train", len(train_ds))
    rec.log_metric("n_val", len(val_ds))

    # DATA_DICT
    import yaml
    dd_path = paths["data_interim"] / "DATA_DICT.yaml"
    with open(dd_path, "r", encoding="utf-8") as f:
        dd = yaml.safe_load(f) or {}
    dd["predict_model"] = {
        "checkpoint": str(ckpt_path.relative_to(paths["root"])),
        "n_params": int(n_params),
        "infer_ms": float(infer_ms),
        "best_val_loss": float(stats["best_val_loss"]),
        "crossing_rate": float(stats["final_crossing_rate"]),
        "normalize_mean": train_ds.normalize["mean"].tolist(),
        "normalize_std": train_ds.normalize["std"].tolist(),
        "architecture": "GRU-D + GRU(hidden=128,2层) + 单调分位数头 + LR调度 + 梯度裁剪",
        "note": "轻量版；GAT 简化为边级 MLP（单边样本无图结构），完整 GAT 在 P10",
    }
    with open(dd_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dd, f, allow_unicode=True, sort_keys=False)

    rec.log_output(str(ckpt_path))
    out = rec.finish(status="success")
    log.info(f"P7 完成: val_loss={stats['best_val_loss']:.4f} crossing={stats['final_crossing_rate']:.4f} "
             f"params={n_params} infer={infer_ms:.2f}ms | {out}")
    log.info("P7 tracking OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
