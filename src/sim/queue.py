"""队列离散更新（技术大纲 §3.2）。

Q_{t+1} = min(B, max(0, Q_t + A_t - S_t))
超 B 或等待超 max_wait 的丢包。

单位：全程 Mb（流量 Mbps × Δt_s）。
"""
from __future__ import annotations

import numpy as np


def capacity_per_slot(capacity_gbps: float, dt_min: float) -> float:
    """链路每时隙可服务量 (Mb)。C(Gbps)×1000(Mbps)×Δt(s)/8 → Mb?
    统一：volume 是 Mbps（速率），时隙内流量 = volume × Δt_s (Mb，因 Mbps×s/8? )
    实际：Mbps × 秒 / 8 = Mb。但为简化，volume×Δt_s 直接作 "Mbps·时隙" 单位，
    容量也用同单位：C_Gbps×1000×Δt_s/8 = Mb。两者一致即可。
    这里用 Mb：volume_mbps × Δt_s / 8 = Mb。
    """
    dt_s = dt_min * 60.0
    return capacity_gbps * 1000.0 * dt_s / 8.0  # Mb


def volume_to_mb(volume_mbps: float, dt_min: float) -> float:
    """速率 Mbps → 时隙内流量 Mb。"""
    return volume_mbps * dt_min * 60.0 / 8.0


class QueueState:
    """单条有向链路的队列状态。"""

    def __init__(self, capacity_mbs: float, buffer_mbs: float, max_wait_slots: int):
        # capacity_mbs: 每时隙服务量(Mb); buffer_mbs: 缓存上限(Mb)
        self.capacity = capacity_mbs
        self.buffer = buffer_mbs
        self.max_wait = max_wait_slots
        self.queue = 0.0           # 当前队列 Mb
        self.queue_age = 0         # 队首已等待时隙数
        # 累计统计
        self.offered = 0.0
        self.served = 0.0
        self.dropped_overflow = 0.0
        self.dropped_timeout = 0.0

    def step(self, arrival_mb: float):
        """一个时隙：到达 arrival_mb，服务，更新队列。

        返回 (served, dropped_this_slot, queue_next, offered_this_slot)。
        offered_this_slot = arrival_mb（本时隙到达量）。
        """
        self.offered += arrival_mb
        offered_slot = arrival_mb
        total = self.queue + arrival_mb
        dropped_overflow_slot = 0.0
        # 缓存溢出：超过 buffer 的立即丢
        if total > self.buffer:
            dropped_overflow_slot = total - self.buffer
            self.dropped_overflow += dropped_overflow_slot
            total = self.buffer
        # 超时丢包：队首等待超 max_wait
        dropped_timeout_slot = 0.0
        if self.queue_age >= self.max_wait and self.queue > 0:
            dropped_timeout_slot = self.queue
            self.dropped_timeout += dropped_timeout_slot
            total -= self.queue
            self.queue = 0.0
            self.queue_age = 0
        # 服务
        served = min(total, self.capacity)
        remaining = total - served
        self.queue = remaining
        self.served += served
        if remaining > 0:
            self.queue_age += 1
        else:
            self.queue_age = 0
        dropped_slot = dropped_overflow_slot + dropped_timeout_slot
        return served, dropped_slot, self.queue, offered_slot

    def reset(self):
        self.queue = 0.0
        self.queue_age = 0
        self.offered = 0.0
        self.served = 0.0
        self.dropped_overflow = 0.0
        self.dropped_timeout = 0.0
