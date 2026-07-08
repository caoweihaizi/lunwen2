"""跨策略与统计审计（§4.7 第5-8项 + §7.2 第1-4,7项）。"""
from __future__ import annotations

import numpy as np

from src.common import resolve_paths


def _load_run_utils(cfg, run_name, sample_shards=30):
    """加载某 run 的利用率/丢包/队列（抽样 shard）。"""
    p = resolve_paths(cfg)
    import glob
    shards = sorted((p["data_processed"] / "sim" / run_name).glob("link_state_shard*.npz"))
    idxs = np.linspace(0, len(shards) - 1, min(sample_shards, len(shards))).astype(int)
    utils = []; queues = []; drops = []
    for i in idxs:
        ls = np.load(shards[i])["link_state"]
        utils.append(ls[:, 7])
        queues.append(ls[:, 5])
        drops.append(ls[:, 6])
    return (np.concatenate(utils), np.concatenate(queues), np.concatenate(drops))


def compare_policies(cfg):
    """三策略 baseline 的利用率/丢包/路径长度对比。"""
    import json
    p = resolve_paths(cfg)
    out = {}
    for pol in ["dijkstra", "ecmp", "queue_aware"]:
        run = f"{pol}_baseline"
        s = json.load(open(p["data_processed"] / "sim" / run / "summary.json"))
        u, q, d = _load_run_utils(cfg, run)
        out[pol] = {
            "util_median": float(np.median(u)),
            "util_p95": float(np.percentile(u, 95)),
            "drop_rate": float(s["drop_rate"]),
            "queue_mean": float(np.mean(q)),
            "e2e_ms": float(s["e2e_mean_ms"]),
        }
    # 三策略产生不同分布：丢包率不全部相同
    drops = [out[pol]["drop_rate"] for pol in out]
    diff = max(drops) - min(drops)
    return {"pass": diff > 0.005, "stats": out, "drop_spread": float(diff)}


def spatial_correlation(cfg, sample_shards=10):
    """链路流量在拓扑邻居 vs 非邻居的相关系数。

    邻居链路（共享节点）的流量相关性应高于非邻居。
    """
    p = resolve_paths(cfg)
    import networkx as nx
    run_dir = p["data_processed"] / "sim" / "ecmp_baseline"
    shards = sorted(run_dir.glob("link_state_shard*.npz"))
    idxs = np.linspace(0, len(shards) - 1, min(sample_shards, len(shards))).astype(int)
    # 累积每条有向边的 carried 时序
    edge_series = {}
    for i in idxs:
        ls = np.load(shards[i])["link_state"]
        for r in ls:
            key = (int(r[1]), int(r[2]))
            edge_series.setdefault(key, []).append(float(r[4]))
    edges = list(edge_series.keys())
    series = [np.array(edge_series[e]) for e in edges]
    n = len(edges)
    # 邻居：共享节点（i,j) 与 (j,k) 或 (k,i)
    def adjacent(e1, e2):
        return bool(set(e1) & set(e2)) and e1 != e2
    adj_corrs = []; nonadj_corrs = []
    min_len = min(len(s) for s in series)
    series = [s[:min_len] for s in series]
    for a in range(n):
        for b in range(a + 1, n):
            if len(series[a]) < 5:
                continue
            c = np.corrcoef(series[a], series[b])[0, 1]
            if np.isnan(c):
                continue
            if adjacent(edges[a], edges[b]):
                adj_corrs.append(c)
            else:
                nonadj_corrs.append(c)
    adj_mean = float(np.mean(adj_corrs)) if adj_corrs else 0
    nonadj_mean = float(np.mean(nonadj_corrs)) if nonadj_corrs else 0
    return {"pass": adj_mean > nonadj_mean, "adj_corr": adj_mean,
            "nonadj_corr": nonadj_mean,
            "n_adj": len(adj_corrs), "n_nonadj": len(nonadj_corrs)}


def load_drop_relation(cfg, sample_shards=20):
    """负载-丢包关系：高负载链路丢包更多。"""
    u, q, d = _load_run_utils(cfg, "ecmp_baseline", sample_shards)
    # 按利用率分桶看丢包
    bins = [0, 0.2, 0.5, 0.8, 1.01]
    rel = []
    for i in range(len(bins) - 1):
        m = (u >= bins[i]) & (u < bins[i + 1])
        if m.any():
            rel.append({"util_bin": f"{bins[i]}-{bins[i+1]}",
                        "mean_drop": float(d[m].mean()),
                        "n": int(m.sum())})
    # 高负载 bin 丢包应更高
    passes = rel[-1]["mean_drop"] > rel[0]["mean_drop"] if len(rel) >= 2 else False
    return {"pass": bool(passes), "bins": rel}


def periodicity_check(cfg):
    """GÉANT/Abilene 自相关周期性复核。"""
    p = resolve_paths(cfg)
    out = {}
    for net in ["geant", "abilene"]:
        S = np.load(p["data_interim"] / net / "S_clean.npz")["S_clean"]
        Sc = S - S.mean()
        var = np.var(Sc)
        acf = np.correlate(Sc, Sc, "full")[len(Sc) - 1:] / (var * len(Sc)) if var > 0 else np.zeros(len(Sc))
        # 找第一个非零 lag 的峰（日周期）
        gran = 15 if net == "geant" else 5
        day_lag = 24 * 60 // gran
        # 在 day_lag 附近找峰
        window = acf[day_lag - 5: day_lag + 5] if len(acf) > day_lag + 5 else acf[-10:]
        peak = float(max(window)) if len(window) else 0
        out[net] = {"day_lag_acf": peak, "granularity_min": gran}
    return {"pass": out["geant"]["day_lag_acf"] > 0.3, "stats": out}
