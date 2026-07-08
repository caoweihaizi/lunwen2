"""P5 阶段入口：遥测缺失观测生成（流式逐 shard）。

对 P4 的 9 run × 4 缺失机制生成观测表。
- MCAR20/MCAR40/MAR20：逐 shard 独立注入（无状态）。
- Block10：预生成全局 block 事件列表（链路, 起始时隙），逐 shard 应用（跨 shard 连续）。
- compound 场景额外生成 compound_mar（= MAR20）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    ExperimentRecord,
    get_logger,
    load_config,
    make_seed_stream,
    resolve_paths,
    seed_everything,
)
from src.missing import build_observed_shard, FIELD_COLS  # noqa: E402
from src.missing.inject import inject_mcar, inject_mar  # noqa: E402
from src.missing.observed import build_observed_shard as _build_obs_shard  # noqa: E402

MECHANISMS = ["mcar20", "mcar40", "mar20", "block10"]
# compound 场景额外生成复合（MAR）
COMPOUND_EXTRA = ["compound_mar"]

FIELD_COL_LIST = list(FIELD_COLS)


def _gen_block_events(edge_set, T, block_len, target_rate, rng):
    """预生成 block 事件：(link_key, start_t) 列表，使总缺失率≈target。

    每事件让 link_key 在 [start_t, start_t+block_len) 缺失。
    """
    n_edges = len(edge_set)
    # 总记录数 = T × n_edges（近似，实际每时隙活跃边数略变）
    # 总缺失 = n_events × block_len，缺失率 = n_events×block_len / (T×n_edges)
    n_events = max(1, int(round(target_rate * T * n_edges / block_len)))
    keys = list(edge_set)
    events = []
    for _ in range(n_events):
        k = keys[rng.randint(len(keys))]
        start = rng.randint(0, max(1, T - block_len))
        events.append((k, start))
    return events, block_len


def _apply_block_to_shard(ls, events, block_len, field_cols):
    """对 shard 应用 block 事件，返回 (mask, observed)。"""
    M = len(ls)
    n_f = len(field_cols)
    mask = np.ones((M, n_f), dtype=bool)
    # 建索引：(i,j,t) -> record idx
    by_edge_t = {}
    for idx in range(M):
        key = (int(ls[idx, 1]), int(ls[idx, 2]))
        t = int(ls[idx, 0])
        by_edge_t[(key, t)] = idx
    for (ek, start) in events:
        for dt in range(block_len):
            t = start + dt
            idx = by_edge_t.get((ek, t))
            if idx is not None:
                mask[idx, :] = False
    observed = ls.astype(np.float64).copy()
    for k, col in enumerate(field_cols):
        observed[~mask[:, k], col] = np.nan
    actual = 1.0 - mask.mean()
    return mask, observed, actual


def _process_run(run_dir, out_base, cfg, log):
    """处理一个 run：4-5 机制 × 114 shard。"""
    run_name = run_dir.name
    shards = sorted(run_dir.glob("link_state_shard*.npz"),
                    key=lambda p: int(p.stem.split("shard")[1]))
    if not shards:
        log.warning(f"{run_name}: 无 shard")
        return None

    # 读第一个 shard 取边集与 T（用于 block 事件）
    ls0 = np.load(shards[0])["link_state"]
    edge_set = set()
    T_max = 0
    for s in shards:
        ls = np.load(s, allow_pickle=True)["link_state"]
        for r in ls:
            edge_set.add((int(r[1]), int(r[2])))
            T_max = max(T_max, int(r[0]))
    T = T_max + 1

    # 每机制独立 generator
    seed_streams = {m: make_seed_stream(cfg.seed.data, f"missing_{run_name}_{m}") for m in MECHANISMS + COMPOUND_EXTRA}
    rngs = {m: np.random.RandomState(next(seed_streams[m])) for m in MECHANISMS + COMPOUND_EXTRA}

    # block10 预生成事件
    block_events, block_len = _gen_block_events(edge_set, T, int(cfg.missing.block_length),
                                                0.20, rngs["block10"])

    # 机制列表：compound run 额外加 compound_mar
    mechs = list(MECHANISMS)
    if run_name.startswith("compound") or run_name.endswith("compound"):
        mechs = MECHANISMS + COMPOUND_EXTRA

    # 逐机制逐 shard
    run_stats = {}
    out_dirs = {}
    for m in mechs:
        od = out_base / f"{run_name}_{m}"
        od.mkdir(parents=True, exist_ok=True)
        out_dirs[m] = od
        run_stats[m] = {"n_shards": 0, "actual_rates": [], "mar_stats": None}

    prev_age = {m: None for m in mechs}  # 跨 shard age 接续

    for si, sp in enumerate(shards):
        ls = np.load(sp, allow_pickle=True)["link_state"]
        for m in mechs:
            if m == "block10":
                mask, obs, actual = _apply_block_to_shard(ls, block_events, block_len, FIELD_COL_LIST)
                age = _compute_age(ls, mask, prev_age[m])
                prev_age[m] = _update_age_state(ls, mask, prev_age[m])
                stats = {"mechanism": "block", "actual_rate": float(actual)}
            elif m == "compound_mar":
                mask, obs, stats = inject_mar(ls, 0.20, rngs[m])
                age = _compute_age(ls, mask, prev_age[m])
                prev_age[m] = _update_age_state(ls, mask, prev_age[m])
            else:
                res = build_observed_shard(ls, m, cfg, rngs[m], prev_age_state=prev_age[m])
                mask, obs, age, stats = res["mask"], res["observed"], res["age"], res["stats"]
                prev_age[m] = res["prev_age_state"]
            run_stats[m]["n_shards"] += 1
            run_stats[m]["actual_rates"].append(float(stats.get("actual_rate", 0)))
            if "high_load_miss" in stats:
                run_stats[m]["mar_stats"] = stats
            # 落盘观测 shard
            np.savez_compressed(out_dirs[m] / f"observed_shard{si}.npz",
                                ls_truth=ls, mask=mask, observed=obs, age=age,
                                field_cols=np.array(FIELD_COL_LIST))

    # 汇总
    summary = {}
    for m, s in run_stats.items():
        rates = s["actual_rates"]
        summary[m] = {
            "mean_actual_rate": float(np.mean(rates)) if rates else 0,
            "n_shards": s["n_shards"],
            "mar_high_load_miss": s["mar_stats"].get("high_load_miss") if s["mar_stats"] else None,
            "mar_low_load_miss": s["mar_stats"].get("low_load_miss") if s["mar_stats"] else None,
        }
    with open(out_base / f"{run_name}_missing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _compute_age(ls, mask, prev_state):
    n_f = mask.shape[1]
    age = np.zeros((len(ls), n_f), dtype=np.int32)
    state = prev_state or {}
    edges = {}
    for idx in range(len(ls)):
        key = (int(ls[idx, 1]), int(ls[idx, 2]))
        edges.setdefault(key, []).append(idx)
    for key, idxs in edges.items():
        last_t = state.get(key, [-1] * n_f)
        for idx in idxs:
            t = int(ls[idx, 0])
            for k in range(n_f):
                if mask[idx, k]:
                    age[idx, k] = 0
                    last_t[k] = t
                else:
                    age[idx, k] = (t - last_t[k]) if last_t[k] >= 0 else (t + 1)
        state[key] = last_t
    return age


def _update_age_state(ls, mask, prev_state):
    n_f = mask.shape[1]
    state = dict(prev_state) if prev_state else {}
    edges = {}
    for idx in range(len(ls)):
        key = (int(ls[idx, 1]), int(ls[idx, 2]))
        edges.setdefault(key, []).append(idx)
    for key, idxs in edges.items():
        last_t = list(state.get(key, [-1] * n_f))
        for idx in idxs:
            t = int(ls[idx, 0])
            for k in range(n_f):
                if mask[idx, k]:
                    last_t[k] = t
        state[key] = last_t
    return state


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.seed.data)
    log = get_logger("p5_main", "P5")
    rec = ExperimentRecord(cfg=cfg, seed=cfg.seed.data, stage="P5")
    paths = resolve_paths(cfg)
    sim_dir = paths["data_processed"] / "sim"
    out_base = paths["data_processed"] / "observed"
    out_base.mkdir(parents=True, exist_ok=True)

    log.info("=== P5 遥测缺失观测生成 ===")

    run_dirs = sorted([d for d in sim_dir.iterdir() if d.is_dir()])
    log.info(f"P4 run 数: {len(run_dirs)}")

    all_summary = {}
    for rd in run_dirs:
        log.info(f"处理 {rd.name} ...")
        s = _process_run(rd, out_base, cfg, log)
        all_summary[rd.name] = s
        if s:
            for m, ms in s.items():
                rec.log_metric(f"{rd.name}_{m}_rate", ms["mean_actual_rate"])

    # DATA_DICT
    import yaml
    dd_path = paths["data_interim"] / "DATA_DICT.yaml"
    with open(dd_path, "r", encoding="utf-8") as f:
        dd = yaml.safe_load(f) or {}
    dd["observed"] = {
        "mechanisms": MECHANISMS + COMPOUND_EXTRA,
        "field_cols": FIELD_COL_LIST,
        "runs": all_summary,
        "note": "流式逐 shard 注入；block 跨 shard 用全局事件列表；真值与观测并存",
    }
    with open(dd_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dd, f, allow_unicode=True, sort_keys=False)

    rec.log_output(str(out_base))
    out = rec.finish(status="success")
    log.info(f"P5 完成，记录落盘: {out}")
    log.info("P5 tracking OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
