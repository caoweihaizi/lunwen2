"""预测训练样本构建（稠密张量版）。

全局有向边并集 ~396 条（Walker 规则星座边集有限）。
每个 (run,mech) 预处理为稠密 (T, n_edges, N_FEAT) + 真值标签 (T, n_edges)。
训练按窗口切片，GPU 友好。

特征（每时隙每边，N_FEAT=12）：
  0 carried, 1 util, 2 queue（observed，缺失处 NaN→0，由 mask 体现）
  3 mask_carried, 4 mask_util, 5 mask_queue
  6 age_carried, 7 age_util, 8 age_queue
  9 delay, 10 dist, 11 active
标签：未来 h 时隙的真值 carried（标准化）
未来计划拓扑：未来 H 时隙该边的 delay/dist/active（可提前使用）
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.common import resolve_paths

COL_T, COL_I, COL_J = 0, 1, 2
COL_CARRIED, COL_QUEUE, COL_UTIL = 4, 5, 7
COL_DELAY, COL_DIST = 8, 9
N_FEAT = 12


def build_global_edge_set(cfg):
    """收集全时隙有向边并集，返回 sorted list + (i,j)->idx 映射。"""
    p = resolve_paths(cfg)
    ts = np.load(p["data_interim"] / "topo" / "topo_series.npz", allow_pickle=True)
    edges = set()
    for el in ts["edge_lists"]:
        for e in el:
            i, j = int(e[0]), int(e[1])
            edges.add((i, j)); edges.add((j, i))
    edge_list = sorted(edges)
    edge_idx = {e: k for k, e in enumerate(edge_list)}
    return edge_list, edge_idx


def preprocess_run(cfg, run, mechanism, edge_idx, cache_path):
    """把 (run,mech) 的 observed + truth 转成稠密张量并缓存。

    保存:
      feat (T, n_edges, N_FEAT) float32（observed 特征，缺失 NaN→0）
      truth (T, n_edges) float32（真值 carried，用于标签）
      mask (T, n_edges, 3)（缺失掩码）
    """
    p = resolve_paths(cfg)
    obs_dir = p["data_processed"] / "observed" / f"{run}_{mechanism}"
    shards = sorted(obs_dir.glob("observed_shard*.npz"),
                    key=lambda x: int(x.stem.split("shard")[1]))
    # 收集全部记录
    ls_list, obs_list, mask_list, age_list = [], [], [], []
    for sp in shards:
        d = np.load(sp, allow_pickle=True)
        ls_list.append(d["ls_truth"])
        obs_list.append(d["observed"])
        mask_list.append(d["mask"])
        age_list.append(d["age"])
    ls = np.concatenate(ls_list)
    obs = np.concatenate(obs_list)
    mask = np.concatenate(mask_list)
    age = np.concatenate(age_list)

    T = int(ls[:, COL_T].max()) + 1
    n_edges = len(edge_idx)
    feat = np.zeros((T, n_edges, N_FEAT), dtype=np.float32)
    truth = np.zeros((T, n_edges), dtype=np.float32)

    for idx in range(len(ls)):
        t = int(ls[idx, COL_T]); i = int(ls[idx, COL_I]); j = int(ls[idx, COL_J])
        k = edge_idx.get((i, j))
        if k is None:
            continue
        # observed 值（缺失 NaN）
        feat[t, k, 0] = obs[idx, COL_CARRIED]  # carried
        feat[t, k, 1] = obs[idx, COL_UTIL]
        feat[t, k, 2] = obs[idx, COL_QUEUE]
        feat[t, k, 3] = mask[idx, 0]  # mask_carried
        feat[t, k, 4] = mask[idx, 1]
        feat[t, k, 5] = mask[idx, 2]
        feat[t, k, 6] = age[idx, 0]
        feat[t, k, 7] = age[idx, 1]
        feat[t, k, 8] = age[idx, 2]
        feat[t, k, 9] = ls[idx, COL_DELAY]
        feat[t, k, 10] = ls[idx, COL_DIST]
        feat[t, k, 11] = 1.0  # active
        # 真值标签
        truth[t, k] = ls[idx, COL_CARRIED]
        # observed 缺失处 NaN→0
        for c in (0, 1, 2):
            if np.isnan(feat[t, k, c]):
                feat[t, k, c] = 0.0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, feat=feat, truth=truth)
    return feat, truth


class LinkStateDataset(Dataset):
    """训练样本：(历史窗口 W, 边) → 未来 H carried。

    从预处理的稠密张量按窗口切片。
    """

    def __init__(self, cfg, run_mechanism_pairs, split_range, n_windows=2000,
                 W=30, H=(1, 3), seed=0, normalize=None, cache_dir=None):
        self.cfg = cfg
        self.W = W
        self.H = H
        self.H_max = max(H)
        p = resolve_paths(cfg)
        self.cache_dir = cache_dir or (p["data_processed"] / "predict" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 全局边集
        self.edge_list, self.edge_idx = build_global_edge_set(cfg)
        self.n_edges = len(self.edge_list)

        # 预处理每个 (run,mech)
        self.tensors = {}  # (run,mech) -> (feat, truth)
        for run, mech in run_mechanism_pairs:
            cache_path = self.cache_dir / f"{run}_{mech}.npz"
            if cache_path.exists():
                d = np.load(cache_path)
                feat, truth = d["feat"], d["truth"]
            else:
                feat, truth = preprocess_run(cfg, run, mech, self.edge_idx, cache_path)
            self.tensors[(run, mech)] = (feat, truth)

        # 采样窗口
        t_start, t_end = split_range
        valid_start = t_start + W
        valid_end = t_end - self.H_max - 1
        self.samples = []
        rng = np.random.RandomState(seed)
        if valid_end > valid_start:
            for run, mech in run_mechanism_pairs:
                for _ in range(n_windows):
                    ts = rng.randint(valid_start, valid_end + 1)
                    # 随机选一条在该窗口活跃的边
                    feat = self.tensors[(run, mech)][0]
                    active_edges = np.where(feat[ts, :, 11] > 0)[0]
                    if len(active_edges) == 0:
                        continue
                    e = active_edges[rng.randint(len(active_edges))]
                    self.samples.append((run, mech, int(e), ts))

        # 标准化（训练集统计）
        if normalize is None:
            self.normalize = self._compute_normalize()
        else:
            self.normalize = normalize

    def _compute_normalize(self):
        vals = []
        for run, mech, e, ts in self.samples[:3000]:
            feat = self.tensors[(run, mech)][0]
            window = feat[ts - self.W:ts, e, :3]  # carried,util,queue
            vals.append(window)
        if not vals:
            return {"mean": np.zeros(3, dtype=np.float32),
                    "std": np.ones(3, dtype=np.float32)}
        vals = np.concatenate(vals)  # (N,3)
        mean = np.nanmean(vals, axis=0)
        std = np.nanstd(vals, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        run, mech, e, ts = self.samples[idx]
        feat, truth = self.tensors[(run, mech)]
        # 历史窗口 (W, N_FEAT)
        X = feat[ts - self.W:ts, e].copy()  # (W, N_FEAT)
        # 标准化前三列
        X[:, :3] = (X[:, :3] - self.normalize["mean"]) / self.normalize["std"]
        # 未来标签 (len(H),) 标准化 carried
        Y = np.zeros(len(self.H), dtype=np.float32)
        for hi, h in enumerate(self.H):
            t_f = ts + h - 1
            if t_f < truth.shape[0]:
                val = truth[t_f, e]
                Y[hi] = (val - self.normalize["mean"][0]) / self.normalize["std"][0]
        # 未来计划拓扑 (H_max, 3): delay, dist, active
        future = np.zeros((self.H_max, 3), dtype=np.float32)
        for k in range(self.H_max):
            t_f = ts + k
            if t_f < feat.shape[0]:
                future[k] = feat[t_f, e, [9, 10, 11]]  # delay, dist, active
        return (torch.from_numpy(X), torch.from_numpy(future),
                torch.from_numpy(Y))
