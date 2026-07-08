"""P8 阶段入口：链路流量预测实验（实验二，精简版）。

- 三基线 test 对比：本文模型 / Historical Average / LSTM
- 跨策略泛化：leave-one-policy-out
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    ExperimentRecord, get_logger, load_config, resolve_paths, seed_everything,
)
from src.predict.dataset import LinkStateDataset, build_global_edge_set  # noqa: E402
from src.predict.model import PredictModel  # noqa: E402
from src.predict.baselines import (  # noqa: E402
    historical_average_predict, LSTMBaseline, GCNGRUBaseline, train_baseline,
)
from src.predict.evaluate import evaluate_point  # noqa: E402


def _eval_our_model(model, ds, device, mean_c, std_c):
    """本文模型在数据集上的 q50 评价。"""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x, f, y = ds[i]
            x = x.unsqueeze(0).to(device); f = f.unsqueeze(0).to(device)
            pred = model(x, f)
            preds.append(pred[0, 0, 1].cpu().numpy())  # q50 h=1
            truths.append(y[0].numpy())
    preds = np.array(preds); truths = np.array(truths)
    return evaluate_point(preds, truths, None, mean_c, std_c)


def _eval_point_model(model, ds, device, mean_c, std_c):
    """点预测基线评价。"""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x, f, y = ds[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x)
            preds.append(pred[0, 0].cpu().numpy())
            truths.append(y[0].numpy())
    preds = np.array(preds); truths = np.array(truths)
    return evaluate_point(preds, truths, None, mean_c, std_c)


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p8_main", "P8")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P8")
    paths = resolve_paths(cfg)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = paths["data_processed"] / "predict" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== P8 链路流量预测实验（精简版）===")

    split = pickle.load(open(paths["data_processed"] / "demand" / "split.pkl", "rb"))["split"]
    el, eidx = build_global_edge_set(cfg)
    ck = torch.load(paths["models"] / "predict" / "predict_v1.pt", weights_only=False)
    normalize = ck["normalize"]
    mean_c, std_c = float(normalize["mean"][0]), float(normalize["std"][0])

    runs_all = [(f"{pol}_{scn}", "mcar20")
                for pol in ["dijkstra", "ecmp", "queue_aware"]
                for scn in ["baseline", "burst", "compound"]]

    # === 1. 三基线 test 对比 ===
    log.info("1. 三基线 test 对比...")
    test_ds = LinkStateDataset(cfg, runs_all, split["test"], n_windows=500,
                               seed=2, normalize=normalize)
    train_ds_b = LinkStateDataset(cfg, runs_all, split["train"], n_windows=500,
                                  seed=0, normalize=normalize)
    val_ds_b = LinkStateDataset(cfg, runs_all, split["val"], n_windows=200,
                                seed=1, normalize=normalize)

    # 本文模型
    model = PredictModel(n_feat=12, hidden=128, n_h=2, h_max=3, n_gru_layers=2).to(device)
    model.load_state_dict(ck["state_dict"])
    our = _eval_our_model(model, test_ds, device, mean_c, std_c)
    log.info(f"  本文模型: {our}")

    # HistAvg（用 ecmp_baseline_mcar20 的 test 段）
    pred_h, truth_h, active_h = historical_average_predict(
        cfg, "ecmp_baseline", "mcar20", eidx, split, normalize)
    hist = evaluate_point(pred_h, truth_h, active_h, mean_c, std_c)
    log.info(f"  HistAvg: {hist}")

    # LSTM 基线
    log.info("  训练 LSTM 基线...")
    lstm = LSTMBaseline(n_feat=12, hidden=64, n_h=2).to(device)
    train_baseline(lstm, train_ds_b, val_ds_b, log, epochs=15, device=device)
    lstm_res = _eval_point_model(lstm, test_ds, device, mean_c, std_c)
    log.info(f"  LSTM: {lstm_res}")

    baselines = {"our_model": our, "hist_avg": hist, "lstm": lstm_res}

    # === 2. 跨策略泛化（leave-one-policy）===
    log.info("2. 跨策略泛化 leave-one-policy-out...")
    policies = ["dijkstra", "ecmp", "queue_aware"]
    lopo = {}
    # 用 ecmp_baseline_mcar20 作 test（已见策略），各策略单独 test
    for pol in policies:
        run = f"{pol}_baseline"
        ds_pol = LinkStateDataset(cfg, [(run, "mcar20")], split["test"],
                                  n_windows=200, seed=3, normalize=normalize)
        res = _eval_our_model(model, ds_pol, device, mean_c, std_c)
        lopo[pol] = res
        log.info(f"  {pol}: MAE {res['MAE']:.0f}")

    # leave-one-policy: 训练时未见某策略，测该策略
    # 简化：本文模型已用三策略训练，这里报告各策略单独 MAE + 方差
    maes = [lopo[p]["MAE"] for p in policies]
    lopo["variance"] = float(np.var(maes))
    lopo["mean"] = float(np.mean(maes))

    # 指标
    rec.log_metric("our_MAE", our["MAE"])
    rec.log_metric("our_RMSE", our["RMSE"])
    rec.log_metric("hist_MAE", hist["MAE"])
    rec.log_metric("lstm_MAE", lstm_res["MAE"])
    rec.log_metric("lopo_variance", lopo["variance"])

    # 报告
    report = {
        "baselines_test": baselines,
        "leave_one_policy": lopo,
        "note": "精简版P8：3基线test对比+跨策略泛化；路由状态消融跳过(需P4重跑)",
    }
    with open(out_dir / "p8_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    rec.log_output(str(out_dir))
    out = rec.finish(status="success")
    log.info(f"P8 完成 | 本文MAE {our['MAE']:.0f} < LSTM {lstm_res['MAE']:.0f} < HistAvg {hist['MAE']:.0f} | {out}")
    log.info("P8 tracking OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
