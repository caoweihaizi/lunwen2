"""P10 阶段入口：路由环境与 MAPPO（分步，先核心）。

轻量版：5000时隙×50epoch，MUCAR vs Point-MAPPO 两基线。
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
from src.data import io  # noqa: E402
from src.route import RouteEnv, Actor, Critic, train_mappo  # noqa: E402


def _load_data(cfg, paths):
    cb = np.load(paths["data_processed"] / "demand" / "commodity_baseline.npz",
                 allow_pickle=True)["commodity"]
    ts = np.load(paths["data_interim"] / "topo" / "topo_series.npz", allow_pickle=True)
    el, ed, de = ts["edge_lists"], ts["edge_dists"], ts["edge_delays"]
    times = io.load_pickle(paths["data_interim"] / "topo" / "times.pkl")["times"]
    p3_k0 = float(json.load(open(paths["data_processed"] / "demand" / "meta.json"))["k0"])
    factor = float(cfg.demand.k0) / p3_k0
    cb = [np.column_stack([a[:, 0], a[:, 1], a[:, 2] * factor, a[:, 3]])
          if len(a) else a for a in cb]
    return cb, el, ed, de, times


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p10_main", "P10")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P10")
    paths = resolve_paths(cfg)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = paths["data_processed"] / "route"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = paths["models"] / "route"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== P10 路由环境与 MAPPO (device={device}) ===")

    # 加载 P7 冻结预测器
    from src.predict.model import PredictModel
    ck = torch.load(paths["models"] / "predict" / "predict_v1.pt", weights_only=False)
    predict_model = PredictModel(n_feat=12, hidden=128, n_h=2, h_max=3, n_gru_layers=2).to(device)
    predict_model.load_state_dict(ck["state_dict"])
    predict_model.eval()
    normalize = ck["normalize"]
    log.info("加载 P7 预测器用于不确定性特征")

    cb, el, ed, de, times = _load_data(cfg, paths)
    # 用 train 段前 5000 时隙
    split = pickle.load(open(paths["data_processed"] / "demand" / "split.pkl", "rb"))["split"]
    train_end = min(split["train"][1], 5000)

    # 评价：训练后在 test 段前 1000 时隙跑
    results = {}

    for tag, use_uncertainty in [("mucar", True), ("point_mappo", False)]:
        log.info(f"\n--- 训练 {tag} (use_uncertainty={use_uncertainty}) ---")
        seed_everything(cfg.seed.data)
        env = RouteEnv(cfg, cb, el, ed, de, times, predict_model=predict_model, normalize=normalize, device=device, max_slots=train_end)
        env.use_uncertainty = use_uncertainty
        env.reset()
        actor = Actor(n_global=3, hidden=64).to(device)
        critic = Critic(n_global_state=env.n_edges * 3, n_sat=env.n_sat, hidden=128, n_global=3).to(device)
        t0 = time.time()
        actor, critic = train_mappo(env, actor, critic, log, epochs=50, n_steps=500,
                                     lr=1e-3, device=device)
        dt = time.time() - t0
        log.info(f"{tag} 训练完成 {dt:.0f}s")

        # 评价（test 段，greedy 动作）
        # 从 test_s - W 开始，前 W 步积累历史（预测器需历史窗口），不记录 reward
        test_s = split["test"][0]
        W = int(cfg.data.history_window_W)
        eval_start = max(0, test_s - W)
        eval_env = RouteEnv(cfg, cb, el, ed, de, times, predict_model=predict_model,
                            normalize=normalize, device=device,
                            max_slots=test_s + 1000)
        eval_env.use_uncertainty = use_uncertainty
        eval_env.reset()
        eval_env.t = eval_start
        # 预热：跑 W 步积累历史，用简单 greedy（valid[0]）不调 actor
        from collections import defaultdict
        from src.sim.potential import all_pairs_hops
        for _ in range(W):
            if eval_env.t >= test_s:
                break
            eval_env.net = eval_env._build_net(eval_env.t)
            hops = all_pairs_hops(eval_env.net.adj)
            comm = eval_env.commodity_ts[eval_env.t]
            agg = defaultdict(float)
            for row in comm:
                s, d, v = int(row[0]), int(row[1]), float(row[2])
                if s != d:
                    agg[(s, d)] += v
            warmup_actions = {}
            for (i, d), vol in agg.items():
                valid = eval_env.get_valid_actions(i, d, hops)
                if valid:
                    warmup_actions[(i, d)] = valid[0]
            eval_env.step(warmup_actions)

        # 正式评价 1000 时隙，用训练好的 actor greedy
        rewards = []; drops = []
        for _ in range(1000):
            if eval_env.t >= eval_env.T:
                break
            eval_env.net = eval_env._build_net(eval_env.t)
            hops = all_pairs_hops(eval_env.net.adj)
            comm = eval_env.commodity_ts[eval_env.t]
            agg = defaultdict(float)
            for row in comm:
                s, d, v = int(row[0]), int(row[1]), float(row[2])
                if s != d:
                    agg[(s, d)] += v
            actions = {}
            globals_list = []; edges_list = []; masks_list = []; keys_list = []
            for (i, d), vol in agg.items():
                g, e, m = eval_env.get_obs(i, d, vol, hops)
                globals_list.append(g); edges_list.append(e); masks_list.append(m)
                keys_list.append((i, d))
            if globals_list:
                g_t = torch.from_numpy(np.stack(globals_list)).to(device)
                e_t = torch.from_numpy(np.stack(edges_list)).to(device)
                m_t = torch.from_numpy(np.stack(masks_list)).to(device)
                with torch.no_grad():
                    logits = actor(g_t, e_t, m_t)
                    action = logits.argmax(dim=1)
                for k, (i, d) in enumerate(keys_list):
                    valid = eval_env.get_valid_actions(i, d, hops)
                    if valid:
                        actions[(i, d)] = valid[int(action[k]) % len(valid)]
            r, done, info = eval_env.step(actions)
            rewards.append(r); drops.append(info["drop_rate"])

        results[tag] = {
            "train_time_s": dt,
            "eval_mean_reward": float(np.mean(rewards)) if rewards else 0,
            "eval_mean_drop": float(np.mean(drops)) if drops else 0,
            "n_eval_steps": len(rewards),
        }
        log.info(f"{tag} eval: reward {results[tag]['eval_mean_reward']:.4f} "
                 f"drop {results[tag]['eval_mean_drop']*100:.2f}%")

        # 落盘 checkpoint
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()},
                   ckpt_dir / f"{tag}.pt")

    # 对比
    log.info(f"\n=== 对比 ===")
    log.info(f"MUCAR reward {results['mucar']['eval_mean_reward']:.4f} vs "
             f"Point-MAPPO {results['point_mappo']['eval_mean_reward']:.4f}")
    log.info(f"MUCAR drop {results['mucar']['eval_mean_drop']*100:.2f}% vs "
             f"Point-MAPPO {results['point_mappo']['eval_mean_drop']*100:.2f}%")

    for tag, r in results.items():
        rec.log_metric(f"{tag}_reward", r["eval_mean_reward"])
        rec.log_metric(f"{tag}_drop", r["eval_mean_drop"])
    rec.log_metric("n_train_slots", train_end)

    with open(out_dir / "p10_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rec.log_output(str(out_dir))
    out = rec.finish(status="success")
    log.info(f"P10 完成 | {out}")
    log.info("P10 tracking OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
