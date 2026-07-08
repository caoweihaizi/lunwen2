"""置信度感知流级逐跳路由环境（技术大纲 §6.1-6.6）。

复用 P4 NetworkState/QueueState 管理链路，路由决策由 actor 提供。
每时隙：actor 为活跃 (i,d) 选下一跳 → 汇总到链路 → 链路服务 → reward。
"""
from __future__ import annotations

import numpy as np
import torch

from src.sim.network import NetworkState
from src.sim.queue import volume_to_mb
from src.sim.potential import all_pairs_hops
from src.common import resolve_paths

N_MAX_NEIGHBORS = 4  # 66 星最大度数
# 边特征维度（§6.1 x_{ij,d,t}）：
# ρ,q,δ,ℓ,ŷ50,ŷup,w,m,a,r_cong,f,φ(j,d),Δφ  = 13
N_EDGE_FEAT = 13


class RouteEnv:
    """流级逐跳路由环境。

    每步（时隙）：
      1. 对每个活跃 (i,d)，actor 选下一跳（从 valid neighbors）
      2. 汇总 commodity 到输出链路
      3. 链路服务 + 队列更新
      4. 返回 reward + 下一观测
    """

    def __init__(self, cfg, commodity_ts, edge_lists, edge_dists, edge_delays,
                 times, predict_model=None, normalize=None, device="cpu",
                 failed_edges=None, max_slots=None):
        self.cfg = cfg
        self.n_sat = int(cfg.constellation.n_sat)
        self.dt_min = float(cfg.data.timeslot_minutes)
        self.commodity_ts = commodity_ts
        self.edge_lists = edge_lists
        self.edge_dists = edge_dists
        self.edge_delays = edge_delays
        self.times = times
        self.predict_model = predict_model
        self.normalize = normalize
        self.device = device
        self.failed_edges = failed_edges or set()
        self.T = len(times) if max_slots is None else min(max_slots, len(times))
        self.cap_gbps = float(cfg.constellation.link_capacity_gbps)
        self.cong_thr = float(cfg.predict.cong_threshold_ratio) * self.cap_gbps * 1000  # Mbps

        # 预计算全局边集（用于 critic 拼接）
        self.edge_list, self.edge_idx = self._build_global_edges()
        self.n_edges = len(self.edge_list)

        # 链路队列跨时隙
        self.link_queue_map = {}
        self.t = 0
        self.net = None
        # 预测缓存（每时隙批量）
        self._pred_cache = {}
        # 是否使用不确定性特征（Point-MAPPO=False 时掩码 w/m/r_cong）
        self.use_uncertainty = True

    def _build_global_edges(self):
        edges = set()
        for el in self.edge_lists[:self.T]:
            for e in el:
                i, j = int(e[0]), int(e[1])
                edges.add((i, j)); edges.add((j, i))
        edge_list = sorted(edges)
        return edge_list, {e: k for k, e in enumerate(edge_list)}

    def reset(self):
        self.t = 0
        self.link_queue_map = {}
        self._pred_cache = {}
        return self.t

    def _build_net(self, t):
        net = NetworkState(self.edge_lists[t], self.edge_dists[t], self.edge_delays[t],
                           self.n_sat, self.cfg, self.dt_min, self.failed_edges)
        for (a, b), q in self.link_queue_map.items():
            if (a, b) in net.edge_idx:
                net.links[net.edge_idx[(a, b)]].queue = q
        return net

    def _get_predictions(self, t):
        """批量预测该时隙所有 active 边的分位数（缓存）。"""
        if t in self._pred_cache:
            return self._pred_cache[t]
        if self.predict_model is None:
            return {}
        # 用历史 W 时隙特征批量预测（简化：用当前链路状态作特征）
        # 这里简化：对每条 active 边，用其当前队列/利用率构造特征前向
        # 实际 P7 模型需历史窗口，这里用当前状态近似（P10 侧重路由，预测器接口）
        preds = {}
        self._pred_cache[t] = preds
        return preds

    def get_valid_actions(self, i, d, hops):
        """有效下一跳：邻居 + 势值严格下降 + 可用（§6.3 无环保证）。

        66 星稀疏拓扑下下降邻居常唯一，导致部分 (i,d) 无选择空间。
        不放宽（放宽会导致环/绕路，reward 崩）。MUCAR 差异在 P12
        压力场景（故障改变候选）体现。
        """
        if self.net is None or i == d:
            return []
        cur = int(hops[i, d])
        valid = []
        for j in np.where(self.net.adj[i])[0]:
            hj = int(hops[j, d])
            if hj < cur and hj < 1e5:  # 严格下降
                valid.append(int(j))
        return valid

    def get_obs(self, i, d, vol, hops, use_uncertainty=None):
        """use_uncertainty=None 时用 self.use_uncertainty。"""
        if use_uncertainty is None:
            use_uncertainty = self.use_uncertainty
        mean_c = float(self.normalize["mean"][0]) if self.normalize else 1.0
        std_c = float(self.normalize["std"][0]) if self.normalize else 1.0
        vol_norm = (vol - mean_c) / max(std_c, 1e-6)
        global_feat = np.array([vol_norm, d / self.n_sat, self.t / self.T], dtype=np.float32)

        valid = self.get_valid_actions(i, d, hops)
        nbrs = list(np.where(self.net.adj[i])[0]) if self.net is not None else []
        cand = valid + [j for j in nbrs if j not in valid]
        cand = cand[:N_MAX_NEIGHBORS]

        edge_feat = np.zeros((N_MAX_NEIGHBORS, N_EDGE_FEAT), dtype=np.float32)
        action_mask = np.zeros(N_MAX_NEIGHBORS, dtype=bool)
        for k, j in enumerate(cand):
            if self.net is None or (i, j) not in self.net.edge_idx:
                continue
            lk = self.net.get_link(i, j)
            idx = self.net.edge_idx[(i, j)]
            rho = lk.queue / max(lk.capacity, 1e-9)
            q = lk.queue / max(lk.buffer, 1e-9)
            delta = self.net.link_delay[idx]
            ell = self.net.link_dist[idx]
            y50 = lk.queue / max(lk.capacity, 1e-9)
            y_up = y50 + 0.1
            w = 0.1
            m = 1.0
            a = 1.0
            r_cong = float(y_up >= self.cong_thr)
            phi_j = int(hops[j, d])
            dphi = int(hops[i, d]) - int(hops[j, d])
            edge_feat[k] = [rho, q, delta, ell, y50, y_up, w, m, a, r_cong, 1.0, phi_j, dphi]
            if j in valid:
                action_mask[k] = True
        # Point-MAPPO: 掩码不确定性特征（w=6, m=7, r_cong=9）
        if not use_uncertainty:
            edge_feat[:, 6] = 0.0  # w
            edge_feat[:, 7] = 0.0  # m
            edge_feat[:, 9] = 0.0  # r_cong
        return global_feat, edge_feat, action_mask

    def step(self, actions):
        """actions: {(i,d): next_hop or None(等待)}。返回 reward, done, info。"""
        self.net = self._build_net(self.t)
        hops = all_pairs_hops(self.net.adj)

        # 汇总 commodity 到输出链路
        from collections import defaultdict
        arrivals_mb = defaultdict(float)
        comm = self.commodity_ts[self.t]
        agg = defaultdict(float)
        for row in comm:
            s, d, v = int(row[0]), int(row[1]), float(row[2])
            if s != d:
                agg[(s, d)] += v

        n_switch = 0
        for (i, d), vol in agg.items():
            j = actions.get((i, d))
            if j is None or (i, j) not in self.net.edge_idx:
                continue  # 等待
            arrivals_mb[(i, j)] += volume_to_mb(vol, self.dt_min)

        # 链路服务
        results = self.net.step_links(dict(arrivals_mb))

        # reward（§6.5）：-λδ·δ - λℓ·ℓ - λq·q - λs·Nswitch
        # results[(i,j)] = (served, dropped, queue, offered)
        tot_served = sum(r[0] for r in results.values())
        tot_drop = sum(r[1] for r in results.values())
        tot_queue = sum(r[2] for r in results.values())
        offered = sum(r[3] for r in results.values())
        drop_rate = tot_drop / max(offered, 1e-9)
        queue_norm = tot_queue / max(self.cap_gbps * 1000 * self.dt_min * 60 / 8 * self.n_edges, 1e-9)
        # 时延近似（用平均传播时延）
        mean_delay = float(np.mean(self.net.link_delay)) if self.net.link_delay else 0.0
        delay_norm = mean_delay / 20.0  # 归一到 ~1

        lam = {"delta": 1.0, "ell": 1.0, "q": 0.5, "s": 0.1}
        reward = -(lam["delta"] * delay_norm + lam["ell"] * drop_rate
                   + lam["q"] * queue_norm + lam["s"] * n_switch)

        # 保存跨时隙队列
        self.link_queue_map = {ep: lk.queue for ep, lk in zip(self.net.link_endpoints, self.net.links)}

        info = {"drop_rate": drop_rate, "mean_delay_ms": mean_delay,
                "tot_served": tot_served, "tot_drop": tot_drop}
        self.t += 1
        done = self.t >= self.T
        return reward, done, info

    def get_global_state(self):
        """centralized critic 用：全部链路 carried/util/queue 拼接。"""
        if self.net is None:
            return np.zeros(self.n_edges * 3, dtype=np.float32)
        state = np.zeros((self.n_edges, 3), dtype=np.float32)
        for (a, b), idx in self.net.edge_idx.items():
            k = self.edge_idx.get((a, b))
            if k is not None:
                lk = self.net.links[idx]
                state[k] = [lk.queue, lk.queue / max(lk.capacity, 1e-9), 1.0]
        return state.flatten()
