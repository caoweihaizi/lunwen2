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
        # 预测缓存（每时隙批量）：{(t, i, j): (q05, q50, q95, w)}
        self._pred_cache = {}
        # 历史特征窗口（每条全局边，W 时隙的 12 维特征）
        self.W = int(cfg.data.history_window_W)
        self._hist = {e: [] for e in self.edge_idx}  # edge -> list of feat(12,)
        # P9 校准器（滑动 CQR，用 calib 集初始化）
        self._scqr = None
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

    def _update_history(self, t):
        """记录该时隙每条 active 边的特征到历史窗口。"""
        if self.net is None:
            return
        mean_c = float(self.normalize["mean"][0]) if self.normalize else 0.0
        std_c = float(self.normalize["std"][0]) if self.normalize else 1.0
        mean_u = float(self.normalize["mean"][1]) if self.normalize else 0.0
        std_u = float(self.normalize["std"][1]) if self.normalize else 1.0
        mean_q = float(self.normalize["mean"][2]) if self.normalize else 0.0
        std_q = float(self.normalize["std"][2]) if self.normalize else 1.0
        for (a, b), idx in self.net.edge_idx.items():
            k = self.edge_idx.get((a, b))
            if k is None:
                continue
            lk = self.net.links[idx]
            carried = (lk.queue - mean_c) / max(std_c, 1e-6)  # 近似：用队列作 carried
            util = lk.queue / max(lk.capacity, 1e-9)
            queue = lk.queue / max(lk.buffer, 1e-9)
            feat = np.array([carried, util, queue, 1, 1, 1, 0, 0, 0,
                             self.net.link_delay[idx], self.net.link_dist[idx], 1.0],
                            dtype=np.float32)
            self._hist[(a, b)].append(feat)
            if len(self._hist[(a, b)]) > self.W:
                self._hist[(a, b)] = self._hist[(a, b)][-self.W:]

    def _get_predictions(self, t):
        """批量预测该时隙所有 active 边的分位数 + 校准区间宽度 w。

        用 P7 模型对历史窗口 W 前向，P9 滑动 CQR 算 w。
        返回 {(i,j): (q05, q50, q95, w)} 标准化空间。
        """
        if t in self._pred_cache:
            return self._pred_cache[t]
        preds = {}
        if self.predict_model is None:
            self._pred_cache[t] = preds
            return preds
        # 收集有足够历史的 active 边
        batch_edges = []; batch_x = []; batch_f = []
        for (a, b), idx in self.net.edge_idx.items():
            hist = self._hist.get((a, b), [])
            if len(hist) < self.W:
                continue
            x = np.stack(hist[-self.W:])  # (W, 12)
            batch_x.append(x)
            # 未来计划（简化：用当前边属性）
            f = np.zeros((3, 3), dtype=np.float32)
            f[:] = [self.net.link_delay[idx], self.net.link_dist[idx], 1.0]
            batch_f.append(f)
            batch_edges.append((a, b))
        if not batch_x:
            self._pred_cache[t] = preds
            return preds
        import torch
        x_t = torch.from_numpy(np.stack(batch_x)).to(self.device)
        f_t = torch.from_numpy(np.stack(batch_f)).to(self.device)
        self.predict_model.eval()
        with torch.no_grad():
            q = self.predict_model(x_t, f_t)  # (B, 2, 3) h=1,3 的 q05/q50/q95
        q = q.cpu().numpy()
        for k, e in enumerate(batch_edges):
            q05, q50, q95 = q[k, 0, 0], q[k, 0, 1], q[k, 0, 2]  # h=1
            # 校准区间宽度 w（简化：q95-q05；P9 滑动 CQR 在 _scqr 维护）
            if self._scqr is not None:
                # 用滑动 q_hat 校准（简化：w = (q95 - q05) + 2*q_hat）
                w = (q95 - q05) + 2 * self._scqr.last_q_hat
            else:
                w = q95 - q05
            preds[e] = (float(q05), float(q50), float(q95), float(w))
        self._pred_cache[t] = preds
        return preds
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

        # 取该时隙预测（批量已算）
        preds = self._get_predictions(self.t)

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
            # 真实预测（从 P7 模型缓存）
            p = preds.get((i, j))
            if p is not None:
                y50, y_up, w = p[1], p[2], p[3]
                m = 1.0  # 观测到（P10 训练假设完整观测）
            else:
                # 历史不足，用当前值兜底
                y50 = lk.queue / max(lk.capacity, 1e-9)
                y_up = y50 + 0.1
                w = 0.1
                m = 0.0  # 标记预测不可用
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
        # 更新历史特征（供下一时隙预测）
        self._update_history(self.t)

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
