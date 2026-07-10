"""多智能体 MAPPO（每个 (i,d) 独立优势）。

标准 MAPPO：共享 actor，centralized critic 对每个 (i,d) 估值。
adv 按每个 (i,d) 序列算，reward 时隙级但 GAE 用每个 (i,d) 的 V。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Actor, Critic
from .env import RouteEnv


def collect_rollout(env, actor, critic, n_steps, device):
    """收集 n_steps 时隙的 rollout，每个 (i,d) 记录独立 V。

    返回 traj：每时隙一个 dict，含该时隙所有 (i,d) 的
    (global, edge, mask, action, logp, cur_sat, dst_sat, value, global_state)
    + 时隙级 reward, done。
    """
    traj = []
    for _ in range(n_steps):
        if env.t >= env.T:
            break
        env.net = env._build_net(env.t)
        from src.sim.potential import all_pairs_hops
        hops = all_pairs_hops(env.net.adj)
        from collections import defaultdict
        comm = env.commodity_ts[env.t]
        agg = defaultdict(float)
        for row in comm:
            s, d, v = int(row[0]), int(row[1]), float(row[2])
            if s != d:
                agg[(s, d)] += v

        if not agg:
            env.t += 1
            traj.append({"reward": 0.0, "done": False, "items": []})
            continue

        # 批量观测
        gl = []; el_ = []; ml = []; kl = []; cur_sats = []; dst_sats = []
        for (i, d), vol in agg.items():
            g, e, m = env.get_obs(i, d, vol, hops)
            gl.append(g); el_.append(e); ml.append(m); kl.append((i, d))
            cur_sats.append(i); dst_sats.append(d)
        g_t = torch.from_numpy(np.stack(gl)).to(device)
        e_t = torch.from_numpy(np.stack(el_)).to(device)
        m_t = torch.from_numpy(np.stack(ml)).to(device)
        cur_t = torch.tensor(cur_sats, dtype=torch.long).to(device)
        dst_t = torch.tensor(dst_sats, dtype=torch.long).to(device)

        with torch.no_grad():
            logits = actor(g_t, e_t, m_t)
            any_valid = m_t.any(dim=1)
            m_safe = m_t.clone(); m_safe[~any_valid] = True
            dist = torch.distributions.Categorical(
                logits=logits.masked_fill(~m_safe, float("-inf")))
            action = dist.sample()
            logp = dist.log_prob(action)
            # critic 对每个 (i,d) 估值（用全局统计量降维，避免 6000 维爆内存）
            gstate_full = env.get_global_state()  # (n_edges*3,)
            # 降维为统计量：均值/P95/max（carried/util/queue 各3）
            gstate = np.array([
                gstate_full[::3].mean(), np.percentile(gstate_full[::3], 95), gstate_full[::3].max(),
                gstate_full[1::3].mean(), np.percentile(gstate_full[1::3], 95), gstate_full[1::3].max(),
                gstate_full[2::3].mean(), gstate_full[2::3].max(),
            ], dtype=np.float32)
            gstate_t = torch.from_numpy(gstate).unsqueeze(0).expand(len(kl), -1).to(device)
            values = critic(gstate_t, cur_t, dst_t, g_t)  # (n_comm,)

        # 构造 actions dict
        actions = {}
        for k, (i, d) in enumerate(kl):
            if any_valid[k]:
                valid = env.get_valid_actions(i, d, hops)
                if valid:
                    actions[(i, d)] = valid[int(action[k]) % len(valid)]

        reward, done, info = env.step(actions)
        traj.append({
            "reward": reward, "done": done,
            "items": [{
                "global": gl[k], "edge": el_[k], "mask": ml[k],
                "action": int(action[k]), "logp": float(logp[k]),
                "cur_sat": cur_sats[k], "dst_sat": dst_sats[k],
                "value": float(values[k]),
            } for k in range(len(kl))],
        })
    return traj


def compute_gae_per_agent(traj, gamma=0.99, lam=0.95):
    """每个 (i,d) 序列独立 GAE。

    简化：用时隙级 reward 作为所有 (i,d) 的共同 reward，
    但用每个 (i,d) 自己的 V 算 adv。这比"全局V"更细粒度。
    """
    T = len(traj)
    # 每时隙每个 item 的 adv
    for t in range(T):
        for item in traj[t]["items"]:
            # adv = reward_t + gamma * V_{t+1} - V_t（简化 TD）
            next_v = traj[t + 1]["items"][0]["value"] if (t + 1 < T and traj[t + 1]["items"]) else 0
            item["adv"] = traj[t]["reward"] + gamma * next_v * (1 - traj[t]["done"]) - item["value"]
            item["ret"] = item["adv"] + item["value"]
    # GAE 平滑（简化：用 TD，不做多步 GAE 回溯，因每时隙 (i,d) 集合不同）
    return traj


def train_mappo(env, actor, critic, log, epochs=20, n_steps=200, lr=1e-3,
                device="mps", clip=0.2, ent_coef=0.01, inner_epochs=2):
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
    for ep in range(epochs):
        traj = collect_rollout(env, actor, critic, n_steps, device)
        if not traj or all(not t["items"] for t in traj):
            continue
        compute_gae_per_agent(traj)

        # 收集所有 items
        all_items = []
        for t in traj:
            for item in t["items"]:
                all_items.append(item)
        if not all_items:
            continue

        # adv 归一化
        advs = np.array([it["adv"] for it in all_items])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        rets = np.array([it["ret"] for it in all_items])

        # 打包成 batch tensor（批量化前向）
        N = len(all_items)
        g_all = torch.from_numpy(np.stack([it["global"] for it in all_items])).to(device)
        e_all = torch.from_numpy(np.stack([it["edge"] for it in all_items])).to(device)
        m_all = torch.from_numpy(np.stack([it["mask"] for it in all_items])).to(device)
        a_all = torch.tensor([it["action"] for it in all_items]).to(device)
        old_logp_all = torch.tensor([it["logp"] for it in all_items]).to(device)
        cur_all = torch.tensor([it["cur_sat"] for it in all_items], dtype=torch.long).to(device)
        dst_all = torch.tensor([it["dst_sat"] for it in all_items], dtype=torch.long).to(device)
        adv_all = torch.from_numpy(advs).float().to(device)
        ret_all = torch.from_numpy(rets).float().to(device)

        total_a = 0.0; total_v = 0.0; n = 0
        batch_size = 256  # 小 batch 避免 critic 全局状态爆内存
        n_gstate = 8  # 全局统计量维度
        for ie in range(inner_epochs):
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                g = g_all[start:end]; e = e_all[start:end]; m = m_all[start:end]
                a = a_all[start:end]; old = old_logp_all[start:end]
                cur = cur_all[start:end]; dst = dst_all[start:end]
                adv = adv_all[start:end]; ret = ret_all[start:end]
                gs = torch.zeros(end - start, n_gstate).to(device)

                # actor 更新
                logits = actor(g, e, m)
                m_safe = m.clone(); m_safe[~m.any(dim=1)] = True
                dist = torch.distributions.Categorical(
                    logits=logits.masked_fill(~m_safe, float("-inf")))
                logp = dist.log_prob(a)
                ratio = torch.exp(logp - old)
                a_loss = -(torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)).mean()
                ent = dist.entropy().mean()
                opt_a.zero_grad(); (a_loss - ent_coef * ent).backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt_a.step()

                # critic 更新
                v = critic(gs, cur, dst, g)
                v_loss = F.mse_loss(v, ret)
                opt_c.zero_grad(); v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_c.step()

                total_a += a_loss.item(); total_v += v_loss.item(); n += 1

        if (ep + 1) % 5 == 0:
            mean_r = np.mean([t["reward"] for t in traj])
            log.info(f"  epoch {ep+1}/{epochs}: a_loss {total_a/max(n,1):.4f} "
                     f"v_loss {total_v/max(n,1):.4f} mean_reward {mean_r:.4f}")
    return actor, critic
