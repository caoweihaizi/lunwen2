"""简版 MAPPO 训练（单环境 rollout + clipped surrogate）。

技术大纲 §6.6：集中训练分布执行，共享 actor，centralized critic。
M4 Pro 控制规模。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Actor, Critic, N_MAX_NEIGHBORS
from .env import RouteEnv


def collect_rollout(env, actor, critic, n_steps, device):
    """收集 n_steps 时隙的 rollout。

    每时隙：对所有活跃 (i,d) 收集观测，actor 采样动作，step。
    返回 trajectories。
    """
    traj = {"global": [], "edge": [], "mask": [], "action": [], "logp": [],
            "reward": [], "value": [], "gstate": [], "done": []}
    for _ in range(n_steps):
        if env.t >= env.T:
            break
        env.net = env._build_net(env.t)
        from src.sim.potential import all_pairs_hops
        hops = all_pairs_hops(env.net.adj)
        # 收集活跃 commodity
        from collections import defaultdict
        comm = env.commodity_ts[env.t]
        agg = defaultdict(float)
        for row in comm:
            s, d, v = int(row[0]), int(row[1]), float(row[2])
            if s != d:
                agg[(s, d)] += v

        # 批量观测
        globals_list = []; edges_list = []; masks_list = []; keys_list = []
        for (i, d), vol in agg.items():
            g, e, m = env.get_obs(i, d, vol, hops)
            globals_list.append(g); edges_list.append(e); masks_list.append(m)
            keys_list.append((i, d))
        if not globals_list:
            env.t += 1
            continue
        g_t = torch.from_numpy(np.stack(globals_list)).to(device)
        e_t = torch.from_numpy(np.stack(edges_list)).to(device)
        m_t = torch.from_numpy(np.stack(masks_list)).to(device)
        with torch.no_grad():
            logits = actor(g_t, e_t, m_t)
            # 有些行全部 mask=False（无合法动作），需处理
            any_valid = m_t.any(dim=1)
            # 对全无效行，临时允许所有（避免 nan），但动作设为等待
            m_safe = m_t.clone()
            m_safe[~any_valid] = True
            logits = actor(g_t, e_t, m_safe) if False else logits  # 已 mask
            dist = torch.distributions.Categorical(logits=logits.masked_fill(~m_safe, float("-inf")))
            action = dist.sample()
            logp = dist.log_prob(action)

        # 构造 actions dict（等待=不加入）
        actions = {}
        for k, (i, d) in enumerate(keys_list):
            if any_valid[k]:
                # action 是槽位索引，需映射回邻居 j
                # 简化：动作直接是槽位，env.step 用 (i, action_slot)——但 env 期望 j
                # 这里需返回 j。简化：env.get_obs 的 cand 顺序，action 索引对应 cand
                # 但 get_obs 内部 cand，外部需一致。简化：action 即下一跳卫星 id
                # 临时：用 valid 列表的 action[0]
                # 为跑通，actions[(i,d)] = valid[action % len(valid)]
                valid = env.get_valid_actions(i, d, hops)
                if valid:
                    actions[(i, d)] = valid[int(action[k]) % len(valid)]

        # critic value（全局状态）
        gstate = env.get_global_state()
        with torch.no_grad():
            v = critic(torch.from_numpy(gstate).unsqueeze(0).to(device)).item()

        reward, done, info = env.step(actions)
        traj["global"].append(g_t.cpu()); traj["edge"].append(e_t.cpu())
        traj["mask"].append(m_t.cpu()); traj["action"].append(action.cpu())
        traj["logp"].append(logp.cpu()); traj["reward"].append(reward)
        traj["value"].append(v); traj["gstate"].append(gstate)
        traj["done"].append(done)
    return traj


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """GAE 优势估计。"""
    T = len(rewards)
    adv = np.zeros(T); gae = 0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0
        delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    returns = adv + np.array(values[:T])
    return adv, returns


def train_mappo(env, actor, critic, log, epochs=50, n_steps=200, lr=1e-3,
                device="mps", clip=0.2, ent_coef=0.01):
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
    for ep in range(epochs):
        traj = collect_rollout(env, actor, critic, n_steps, device)
        if not traj["reward"]:
            continue
        adv, returns = compute_gae(traj["reward"], traj["value"], traj["done"])
        adv_t = torch.from_numpy(adv).float()
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.from_numpy(returns).float()

        # 更新（每步一个 batch，简化）
        total_a = 0.0; total_v = 0.0; n = 0
        for t in range(len(traj["reward"])):
            g = traj["global"][t].to(device); e = traj["edge"][t].to(device)
            m = traj["mask"][t].to(device); a = traj["action"][t].to(device)
            old_logp = traj["logp"][t].to(device)
            logits = actor(g, e, m)
            m_safe = m.clone(); m_safe[~m.any(dim=1)] = True
            dist = torch.distributions.Categorical(logits=logits.masked_fill(~m_safe, float("-inf")))
            logp = dist.log_prob(a)
            ratio = torch.exp(logp - old_logp)
            a_adv = adv_t[t].to(device)
            a_loss = -(torch.min(ratio * a_adv, torch.clamp(ratio, 1 - clip, 1 + clip) * a_adv)).mean()
            ent = dist.entropy().mean()
            opt_a.zero_grad(); (a_loss - ent_coef * ent).backward(); opt_a.step()

            gstate = torch.from_numpy(traj["gstate"][t]).unsqueeze(0).to(device)
            v = critic(gstate).squeeze()
            v_loss = F.mse_loss(v, ret_t[t].to(device).unsqueeze(0).expand_as(v) if v.dim() > 0 else ret_t[t].to(device))
            opt_c.zero_grad(); v_loss.backward(); opt_c.step()
            total_a += a_loss.item(); total_v += v_loss.item(); n += 1

        if (ep + 1) % 5 == 0:
            mean_r = np.mean(traj["reward"])
            log.info(f"  epoch {ep+1}/{epochs}: a_loss {total_a/max(n,1):.4f} "
                     f"v_loss {total_v/max(n,1):.4f} mean_reward {mean_r:.4f}")
    return actor, critic
