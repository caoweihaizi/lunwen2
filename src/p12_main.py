"""P12 阶段入口：压力测试（实验五）。

S1-S5 场景，MUCAR vs Point-MAPPO 的 P95/P99 时延和丢包。
重点 S5 复合压力（MAR+突发+故障）。
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    ExperimentRecord, get_logger, load_config, make_seed_stream,
    resolve_paths, seed_everything,
)
from src.data import io  # noqa: E402
from src.predict.model import PredictModel  # noqa: E402
from src.route import RouteEnv, Actor  # noqa: E402
from src.sim.failure import inject_failures  # noqa: E402
from src.sim.potential import all_pairs_hops  # noqa: E402


def _load_commodity(paths, cfg, scenario="baseline"):
    cb = np.load(paths["data_processed"] / "demand" / f"commodity_{scenario}.npz",
                 allow_pickle=True)["commodity"]
    p3_k0 = float(json.load(open(paths["data_processed"] / "demand" / "meta.json"))["k0"])
    factor = float(cfg.demand.k0) / p3_k0
    return [np.column_stack([a[:, 0], a[:, 1], a[:, 2] * factor, a[:, 3]])
            if len(a) else a for a in cb]


def _eval_scenario(cfg, paths, actor, predict_model, normalize, device, tag,
                   commodity, failed_edges, n_eval=500, log=None):
    """跑一个场景的 eval，返回指标。"""
    split = pickle.load(open(paths["data_processed"] / "demand" / "split.pkl", "rb"))["split"]
    test_s = split["test"][0]
    W = int(cfg.data.history_window_W)
    eval_start = max(0, test_s - W)
    ts = np.load(paths["data_interim"] / "topo" / "topo_series.npz", allow_pickle=True)
    el, ed, de = ts["edge_lists"], ts["edge_dists"], ts["edge_delays"]
    times = io.load_pickle(paths["data_interim"] / "topo" / "times.pkl")["times"]

    env = RouteEnv(cfg, commodity, el, ed, de, times,
                   predict_model=predict_model, normalize=normalize,
                   device=device, failed_edges=failed_edges,
                   max_slots=test_s + n_eval)
    env.use_uncertainty = (tag == "mucar")
    env.reset()
    env.t = eval_start

    # 预热 W 步
    for _ in range(W):
        if env.t >= test_s:
            break
        env.net = env._build_net(env.t)
        hops = all_pairs_hops(env.net.adj)
        comm = commodity[env.t]; agg = defaultdict(float)
        for row in comm:
            s, d, v = int(row[0]), int(row[1]), float(row[2])
            if s != d:
                agg[(s, d)] += v
        act = {}
        for (i, d) in agg:
            v = env.get_valid_actions(i, d, hops)
            if v:
                act[(i, d)] = v[0]
        env.step(act)

    # 正式 eval
    rewards = []; drops = []; delays = []
    for _ in range(n_eval):
        if env.t >= env.T:
            break
        env.net = env._build_net(env.t)
        hops = all_pairs_hops(env.net.adj)
        comm = commodity[env.t]; agg = defaultdict(float)
        for row in comm:
            s, d, v = int(row[0]), int(row[1]), float(row[2])
            if s != d:
                agg[(s, d)] += v
        actions = {}
        gl = []; el_ = []; ml = []; kl = []
        for (i, d), vol in agg.items():
            g, e, m = env.get_obs(i, d, vol, hops)
            gl.append(g); el_.append(e); ml.append(m); kl.append((i, d))
        if gl:
            g_t = torch.from_numpy(np.stack(gl)).to(device)
            e_t = torch.from_numpy(np.stack(el_)).to(device)
            m_t = torch.from_numpy(np.stack(ml)).to(device)
            with torch.no_grad():
                logits = actor(g_t, e_t, m_t)
                action = logits.argmax(dim=1)
            for k, (i, d) in enumerate(kl):
                v = env.get_valid_actions(i, d, hops)
                if v:
                    actions[(i, d)] = v[int(action[k]) % len(v)]
        r, done, info = env.step(actions)
        rewards.append(r); drops.append(info["drop_rate"])
        delays.append(info["mean_delay_ms"])

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0,
        "mean_drop": float(np.mean(drops)) if drops else 0,
        "p95_drop": float(np.percentile(drops, 95)) if drops else 0,
        "p99_drop": float(np.percentile(drops, 99)) if drops else 0,
        "mean_delay": float(np.mean(delays)) if delays else 0,
        "p95_delay": float(np.percentile(delays, 95)) if delays else 0,
        "p99_delay": float(np.percentile(delays, 99)) if delays else 0,
        "n_steps": len(rewards),
    }


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p12_main", "P12")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P12")
    paths = resolve_paths(cfg)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = paths["data_processed"] / "route"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== P12 压力测试 ===")

    # 加载预测器
    ck_pred = torch.load(paths["models"] / "predict" / "predict_v1.pt", weights_only=False)
    predict_model = PredictModel(n_feat=12, hidden=128, n_h=2, h_max=3, n_gru_layers=2).to(device)
    predict_model.load_state_dict(ck_pred["state_dict"]); predict_model.eval()
    normalize = ck_pred["normalize"]

    # 加载两基线 actor
    actors = {}
    for tag in ["mucar", "point_mappo"]:
        ck = torch.load(paths["models"] / "route" / f"{tag}.pt", weights_only=False)
        a = Actor(n_global=3, hidden=64).to(device)
        a.load_state_dict(ck["actor"]); a.eval()
        actors[tag] = a

    # 故障集（S5 用）
    ts = np.load(paths["data_interim"] / "topo" / "topo_series.npz", allow_pickle=True)
    rng_fail = np.random.RandomState(next(make_seed_stream(cfg.seed.data, "failure")))
    failed = inject_failures(ts["edge_lists"], cfg, rng_fail, "compound")
    log.info(f"S5 故障: {len(failed)} 条链路")

    # 场景定义
    cb_base = _load_commodity(paths, cfg, "baseline")
    cb_burst = _load_commodity(paths, cfg, "burst")
    scenarios = [
        ("S1", cb_base, set()),           # 正常
        ("S5", cb_burst, failed),         # 复合压力
    ]

    results = {}
    for scn_name, cb, fail_edges in scenarios:
        log.info(f"\n--- 场景 {scn_name} ---")
        for tag in ["mucar", "point_mappo"]:
            t0 = time.time()
            r = _eval_scenario(cfg, paths, actors[tag], predict_model, normalize,
                               device, tag, cb, fail_edges, n_eval=500, log=log)
            r["eval_time_s"] = round(time.time() - t0, 1)
            results[f"{scn_name}_{tag}"] = r
            log.info(f"  {tag}: reward {r['mean_reward']:.4f} drop {r['mean_drop']*100:.2f}% "
                     f"P95drop {r['p95_drop']*100:.2f}% delay {r['mean_delay']:.1f}ms "
                     f"P95delay {r['p95_delay']:.1f}ms ({r['eval_time_s']}s)")
            rec.log_metric(f"{scn_name}_{tag}_drop", r["mean_drop"])
            rec.log_metric(f"{scn_name}_{tag}_p95delay", r["p95_delay"])

    # 对比
    log.info(f"\n=== 对比 ===")
    for scn in ["S1", "S5"]:
        m = results[f"{scn}_mucar"]; p = results[f"{scn}_point_mappo"]
        log.info(f"{scn}: MUCAR drop {m['mean_drop']*100:.2f}% vs Point {p['mean_drop']*100:.2f}% | "
                 f"MUCAR P95delay {m['p95_delay']:.1f} vs Point {p['p95_delay']:.1f}")

    with open(out_dir / "p12_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rec.log_output(str(out_dir))
    out = rec.finish(status="success")
    log.info(f"P12 完成 | {out}")
    log.info("P12 tracking OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
