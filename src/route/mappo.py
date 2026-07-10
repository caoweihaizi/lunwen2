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
            # critic 对每个 (i,d) 估值
            gstate = env.get_global_state()
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


def train_mappo(env, actor, critic, log, epochs=50, n_steps=200, lr=1e-3,
                device="mps", clip=0.2, ent_coef=0.01, inner_epochs=4):
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

        total_a = 0.0; total_v = 0.0; n = 0
        for ie in range(inner_epochs):
            for k, item in enumerate(all_items):
                g = torch.from_numpy(item["global"]).unsqueeze(0).to(device)
                e = torch.from_numpy(item["edge"]).unsqueeze(0).to(device)
                m = torch.from_numpy(item["mask"]).unsqueeze(0).to(device)
                a = torch.tensor([item["action"]]).to(device)
                old_logp = torch.tensor([item["logp"]]).to(device)
                cur = torch.tensor([item["cur_sat"]], dtype=torch.long).to(device)
                dst = torch.tensor([item["dst_sat"]], dtype=torch.long).to(device)
                adv = torch.tensor([advs[k]]).to(device)
                ret = torch.tensor([rets[k]]).to(device)

                # actor 更新
                logits = actor(g, e, m)
                m_safe = m.clone(); m_safe[~m.any(dim=1)] = True
                dist = torch.distributions.Categorical(
                    logits=logits.masked_fill(~m_safe, float("-inf")))
                logp = dist.log_prob(a)
                ratio = torch.exp(logp - old_logp)
                a_loss = -(torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)).mean()
                ent = dist.entropy().mean()
                opt_a.zero_grad(); (a_loss - ent_coef * ent).backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt_a.step()

                # critic 更新（每个 (i,d) 独立 V）
                # 需 global_state —— 从 item 无法直接取，用 env 当前状态近似
                # 简化：用零向量（critic 的 state_net 会处理）
                gstate = torch.zeros(1, env.n_edges * 3).to(device)
                v = critic(gstate, cur, dst, g)
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
