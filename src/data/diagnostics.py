"""数据诊断图。

为 P6 数据可信度验证提供可视化证据。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_s_timeseries(S, times, title: str, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, S, linewidth=0.5)
    ax.set_title(f"{title}: S(t) global intensity time series")
    ax.set_ylabel("Mbps")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_diurnal_boxplot(S, times, granularity_min: int, title: str, out_path: Path) -> Path:
    """按一天内的时间槽分组，画日周期箱线图。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slots_per_day = int(24 * 60 / granularity_min)
    # 每个时刻在一天中的槽位
    slot_of_day = np.array(
        [(t.hour * 60 + t.minute) // granularity_min for t in times]
    )
    groups = [S[slot_of_day == s] for s in range(slots_per_day)]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.boxplot(groups, positions=range(slots_per_day), showfliers=False, widths=0.7)
    xticks = np.arange(0, slots_per_day, slots_per_day // 8)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(s*granularity_min//60):02d}:00" for s in xticks])
    ax.set_title(f"{title}: diurnal cycle boxplot by time-of-day slot")
    ax.set_ylabel("Mbps")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_autocorr(S, max_lag: int, title: str, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Sc = S - S.mean()
    var = np.var(Sc)
    acf = (
        np.correlate(Sc, Sc, "full")[len(Sc) - 1 :]
        / (var * len(Sc))
        if var > 0
        else np.zeros(len(Sc))
    )
    lags = np.arange(0, max_lag + 1)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(lags, acf[: max_lag + 1], linewidth=0.8)
    ax.set_title(f"{title}: autocorrelation")
    ax.set_xlabel("lag (slots)")
    ax.set_ylabel("ACF")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_resample_spectrum(S_before, before_min: int, S_after, after_min: int,
                           title: str, out_path: Path) -> Path:
    """重采样前后功率谱密度对比，确认不引入虚假高频。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))

    def spec(x, dt_min):
        x = x - x.mean()
        n = len(x)
        freqs = np.fft.rfftfreq(n, d=dt_min)  # cycles/min
        psd = np.abs(np.fft.rfft(x)) ** 2
        return freqs, psd

    f1, p1 = spec(S_before, before_min)
    f2, p2 = spec(S_after, after_min)
    # 归一化便于对比
    ax.semilogy(f1, p1 / p1.max(), label=f"original {before_min}min", alpha=0.8, linewidth=0.6)
    ax.semilogy(f2, p2 / p2.max(), label=f"resampled {after_min}min", alpha=0.8, linewidth=0.6)
    ax.set_title(f"{title}: power spectrum: before vs after resample")
    ax.set_xlabel("frequency (cycles/min)")
    ax.set_ylabel("normalized PSD")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
