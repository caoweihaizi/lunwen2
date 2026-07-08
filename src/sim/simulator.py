"""流级逐跳仿真器主循环（技术大纲 §3.5）。

流式落盘：每 flush_every 时隙把累积记录 flush 到磁盘分片并清空，
避免全段累积导致内存膨胀（P4 v1 因 7255 万明细记录触发 52GB swap）。

默认只存 link_state（P7/P9 训练与评价所需）。明细表（node_commodity/
link_commodity）通过 keep_detail=True 生成，但建议仅在审计时对抽样时段启用。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .network import NetworkState
from .queue import volume_to_mb
from .potential import all_pairs_hops


class FlowLevelSimulator:
    def __init__(self, cfg, dt_min):
        self.cfg = cfg
        self.dt_min = dt_min
        self.n_sat = int(cfg.constellation.n_sat)

    def run(self, commodity_ts, edge_lists, edge_dists, edge_delays, times,
            policy, failed_edges, rng=None, max_slots=None,
            flush_callback=None, flush_every=500, keep_detail=False):
        """跑一次仿真（流式落盘）。

        flush_callback(shard_idx, link_state_chunk, node_comm_chunk,
                       link_comm_chunk, e2e_chunk, t_range): 每 flush_every 时隙调用。
            调用方负责把 chunk 落盘。若为 None，则全量累积返回（兼容旧接口，慎用）。
        keep_detail: True 才记录 node_commodity/link_commodity 明细（内存大头）。
        """
        T = len(times) if max_slots is None else min(max_slots, len(times))
        n = self.n_sat

        link_state_buf = []
        node_comm_buf = [] if keep_detail else None
        link_comm_buf = [] if keep_detail else None
        e2e_buf = []
        shard_idx = 0
        t_start = 0

        # 累计统计（不随 flush 清空，用于最终 summary）
        tot_offered = 0.0
        tot_carried = 0.0
        tot_drop = 0.0
        all_utils = []  # 仅统计用，flush 时不清空（用于最终分位）

        link_queue_map = {}

        for t in range(T):
            edges = edge_lists[t]
            if len(edges) == 0:
                continue
            net = NetworkState(edges, edge_dists[t], edge_delays[t], n,
                               self.cfg, self.dt_min, failed_edges)
            for (a, b), q in link_queue_map.items():
                if (a, b) in net.edge_idx:
                    net.links[net.edge_idx[(a, b)]].queue = q

            policy.prepare(net, None)

            comm = commodity_ts[t]
            if len(comm) == 0:
                results = net.step_links({})
                self._record(link_state_buf, net, t, results, None)
                e2e_buf.append(0.0)
                if (t + 1) % flush_every == 0:
                    self._flush(flush_callback, shard_idx, t_start, t + 1,
                                link_state_buf, node_comm_buf, link_comm_buf, e2e_buf)
                    shard_idx += 1; t_start = t + 1
                continue

            agg = defaultdict(float)
            for row in comm:
                s, d, v = int(row[0]), int(row[1]), float(row[2])
                if s != d:
                    agg[(s, d)] += v
            commodity_active = list(agg.keys())

            if policy.name == "queue_aware":
                decisions = policy.decide(net, commodity_active, t, rng)
            else:
                decisions = policy.decide(net, commodity_active, t)

            arrivals_mb = defaultdict(float)
            link_comm_dict = defaultdict(float) if keep_detail else None
            for (i, d), vol_mbps in agg.items():
                split = decisions.get((i, d), {})
                if not split:
                    if keep_detail:
                        node_comm_buf.append((t, i, d, vol_mbps, -1, 0.0))
                    continue
                for j, r in split.items():
                    v_mb = volume_to_mb(vol_mbps * r, self.dt_min)
                    arrivals_mb[(i, j)] += v_mb
                    if keep_detail:
                        link_comm_dict[(i, j, d)] += v_mb
                        node_comm_buf.append((t, i, d, vol_mbps * r, j, r))

            results = net.step_links(dict(arrivals_mb))

            # 记录链路状态 + 累计统计
            for (a, b), idx in net.edge_idx.items():
                lk = net.links[idx]
                served, dropped, q, offered = results.get((a, b), (0.0, 0.0, lk.queue, 0.0))
                util = served / max(lk.capacity, 1e-9)
                link_state_buf.append((t, a, b, offered, served, q, dropped,
                                       util, net.link_delay[idx], net.link_dist[idx]))
                tot_offered += offered
                tot_carried += served
                tot_drop += dropped
                all_utils.append(util)
            if keep_detail:
                for (i, j, d), load in link_comm_dict.items():
                    link_comm_buf.append((t, i, j, d, load))

            # e2e
            hops = getattr(policy, "hops", None)
            if hops is None:
                hops = all_pairs_hops(net.adj)
            mean_delay = 0.0; cnt = 0
            mean_ld = float(np.mean(net.link_delay)) if net.link_delay else 0.0
            for (i, d), vol in agg.items():
                if hops[i, d] < 1e5:
                    mean_delay += hops[i, d] * mean_ld; cnt += 1
            e2e_buf.append(mean_delay / max(cnt, 1))

            link_queue_map = {ep: lk.queue for ep, lk in zip(net.link_endpoints, net.links)}

            if (t + 1) % flush_every == 0:
                self._flush(flush_callback, shard_idx, t_start, t + 1,
                            link_state_buf, node_comm_buf, link_comm_buf, e2e_buf)
                shard_idx += 1; t_start = t + 1

        # 尾部分片
        if link_state_buf or e2e_buf:
            self._flush(flush_callback, shard_idx, t_start, T,
                        link_state_buf, node_comm_buf, link_comm_buf, e2e_buf)
            shard_idx += 1

        return {
            "n_slots": T,
            "n_shards": shard_idx,
            "tot_offered": tot_offered,
            "tot_carried": tot_carried,
            "tot_drop": tot_drop,
            "all_utils": np.array(all_utils),
            "e2e_mean_ms": float(np.mean(e2e_buf)) if e2e_buf else 0.0,
        }

    def _flush(self, cb, shard_idx, t_start, t_end, ls, nc, lc, e2e):
        if cb is not None:
            cb(shard_idx, ls, nc, lc, e2e, (t_start, t_end))
        # 清空 buffer（nc/lc 若 None 跳过）
        ls.clear()
        if nc is not None:
            nc.clear()
        if lc is not None:
            lc.clear()
        e2e.clear()
