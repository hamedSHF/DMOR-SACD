"""Agents: SACD (DMoR-SACD + ablation variants), DQN-R/DDQN, Rainbow.

Common training scheme (Algorithm 1 of the paper):
    1. sample a batch of users -> run a K-step episode for each
    2. store (s_t, a_t, r_t, s_{t+1}) in the replay buffer
    3. update critics (Eq. 14), actor (Eq. 15), temperature alpha (Eq. 16)
    4. soft-update the target critics (tau = 0.01)

In training actions are sampled from the actor's categorical distribution;
in evaluation the item with maximum probability / Q-value is chosen
(Algorithm 2), and already-recommended items are masked out.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .buffers import PrioritizedReplayBuffer, ReplayBuffer
from .env import EpisodeState, RecommendationEnv
from .models import DQNNetwork, RainbowNetwork, SACDNetwork


@dataclass
class EpisodeResult:
    user: int
    seq: List[int] = field(default_factory=list)
    racc: List[float] = field(default_factory=list)   # normalised ratings
    total: List[float] = field(default_factory=list)  # training rewards


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mask_remaining(eps: List[EpisodeState], n_actions: int) -> np.ndarray:
    """1.0 for items not yet recommended in the episode, else 0.0."""
    mask = np.ones((len(eps), n_actions), dtype=np.float32)
    for i, ep in enumerate(eps):
        if ep.seq:
            mask[i, ep.seq] = 0.0
    return mask


def _push(buf, obs, actions, rewards, next_obs, dones) -> None:
    for i in range(len(actions)):
        buf.push((
            obs[0][i], obs[1][i], obs[2][i],
            int(actions[i]), float(rewards[i]),
            next_obs[0][i], next_obs[1][i], next_obs[2][i],
            bool(dones[i]),
        ))


# ---------------------------------------------------------------------------
# SACD agent (DMoR-SACD, Model 1, 2, 3, 3-sat, 3-div)
# ---------------------------------------------------------------------------
class SACDAgent:
    name = "SACD"

    def __init__(self, cfg, data, env):
        self.cfg = cfg
        self.data = data
        self.env = env
        self.device = torch.device(cfg.device)
        self.n_actions = data.num_items
        self.net = SACDNetwork(data.num_items, self.n_actions, cfg).to(self.device)

        params_actor = list(self.net.actor.parameters()) + list(self.net.actor_encoder.parameters())
        params_critic = (list(self.net.critic1.parameters())
                         + list(self.net.critic2.parameters()))
        self.opt_actor = torch.optim.Adam(params_actor, lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(params_critic, lr=cfg.lr)
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        # Eq. 16 target entropy: H_bar = -0.8 * log(1/|A|)  (paper, Sec. 4.1)
        self.target_entropy = float(
            -cfg.target_entropy_rate * np.log(1.0 / self.n_actions))

        self.buffer = ReplayBuffer(cfg.replay_capacity)
        self.step_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    # -- action selection --------------------------------------------------
    def select_actions(self, obs, eps, deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            probs = self.net.actor_probs(obs)
            mask = torch.as_tensor(_mask_remaining(eps, self.n_actions),
                                   device=self.device)
            masked = probs * mask
            masked = masked / masked.sum(-1, keepdim=True).clamp_min(1e-8)
            if deterministic:
                act = masked.argmax(-1)
            else:
                act = torch.multinomial(masked, 1).squeeze(-1)
        return act.cpu().numpy()

    # -- episode collection ------------------------------------------------
    def collect(self, users: np.ndarray) -> Dict[str, float]:
        eps = self.env.reset_batch(users)
        dev = self.device
        returns, totals, accs = [], [], []
        for _ in range(self.cfg.interaction_length):
            obs = self.env.build_obs(eps, dev)
            obs_np = [t.cpu().numpy() for t in obs]
            actions = self.select_actions(obs, eps, deterministic=False)
            rewards, dones = [], []
            for ep, a in zip(eps, actions):
                rew = self.env.step(ep, int(a))
                rewards.append(rew["total"])
                accs.append(rew["racc"])
                dones.append(self.env.done(ep))
            next_obs = self.env.build_obs(eps, dev)
            _push(self.buffer, obs_np, actions, rewards,
                  [t.cpu().numpy() for t in next_obs], dones)
            returns.append(float(np.sum(rewards)))
            totals.append(float(np.mean(rewards)))
            if all(dones):
                break
        return {"episode_return": float(np.mean(returns)),
                "mean_step_reward": float(np.mean(totals))}

    # -- gradient update ---------------------------------------------------
    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {}
        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        obs = (batch["pos"], batch["neg"], batch["seq"])
        obs2 = (batch["next_pos"], batch["next_neg"], batch["next_seq"])
        a, r, done = batch["action"], batch["reward"], batch["done"]

        # --- critics (Eq. 14) ---
        with torch.no_grad():
            probs2 = self.net.actor_probs(obs2).clamp_min(1e-8)
            logp2 = probs2.log()
            q_t = self.net.target_values(obs2)                  # min(Q'1, Q'2)
            v_t = (probs2 * (q_t - self.alpha * logp2)).sum(-1)  # Eq. 12
            y = r + self.cfg.gamma * (1 - done) * v_t
        q1 = self.net.critic_values(1, obs).gather(1, a.unsqueeze(-1)).squeeze(-1)
        q2 = self.net.critic_values(2, obs).gather(1, a.unsqueeze(-1)).squeeze(-1)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad()
        loss_q.backward()
        self.opt_critic.step()

        # --- actor (Eq. 15) ---
        probs = self.net.actor_probs(obs).clamp_min(1e-8)
        logp = probs.log()
        with torch.no_grad():
            q_min = self.net.min_critic(obs)
        loss_pi = (probs * (self.alpha * logp - q_min)).sum(-1).mean()
        self.opt_actor.zero_grad()
        loss_pi.backward()
        self.opt_actor.step()

        # --- temperature (Eq. 16) ---
        logp_a = logp.gather(1, a.unsqueeze(-1)).squeeze(-1)
        alpha_loss = -(self.log_alpha * (logp_a.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        self.net.soft_update(self.cfg.tau)
        self.step_count += 1
        return {"critic_loss": float(loss_q.item()),
                "actor_loss": float(loss_pi.item()),
                "alpha": float(self.alpha.item())}

    # -- evaluation --------------------------------------------------------
    def evaluate(self, users: np.ndarray) -> List[EpisodeResult]:
        self.net.eval()
        eps = self.env.reset_batch(users)
        dev = self.device
        results = [EpisodeResult(user=int(u)) for u in users]
        for _ in range(self.cfg.interaction_length):
            obs = self.env.build_obs(eps, dev)
            actions = self.select_actions(obs, eps, deterministic=True)
            for i, (ep, a) in enumerate(zip(eps, actions)):
                rew = self.env.step(ep, int(a))
                results[i].seq.append(int(a))
                results[i].racc.append(rew["racc"])
                results[i].total.append(rew["total"])
            if all(self.env.done(ep) for ep in eps):
                break
        self.net.train()
        return results

    # -- warm-start checkpoints --------------------------------------------
    def state_dict(self, include_buffer: bool = False) -> dict:
        """Full trainer state for exact warm start."""
        d = {
            "agent": self.name,
            "net": self.net.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "opt_alpha": self.opt_alpha.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().clone(),
            "step_count": self.step_count,
            "buffer": None,
        }
        if include_buffer:
            d["buffer"] = list(self.buffer.buf)
        return d

    def load_state_dict(self, d: dict) -> None:
        """Restore a checkpoint produced by ``state_dict``."""
        if d.get("agent") and d["agent"] != self.name:
            print(f"[warn] checkpoint agent '{d['agent']}' does not match "
                  f"'{self.name}'; loading anyway")
        self.net.load_state_dict(d["net"])
        self.opt_actor.load_state_dict(d["opt_actor"])
        self.opt_critic.load_state_dict(d["opt_critic"])
        self.opt_alpha.load_state_dict(d["opt_alpha"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))
        self.step_count = d.get("step_count", 0)
        buf = d.get("buffer")
        if buf:
            self.buffer.buf = deque(buf, maxlen=self.cfg.replay_capacity)


# ---------------------------------------------------------------------------
# DQN agent (DQN-R with GRU state, DDQN with plain state)
# ---------------------------------------------------------------------------
class DQNAgent:
    name = "DQN"

    def __init__(self, cfg, data, env):
        self.cfg = cfg
        self.data = data
        self.env = env
        self.device = torch.device(cfg.device)
        self.n_actions = data.num_items
        self.net = DQNNetwork(data.num_items, self.n_actions, cfg).to(self.device)
        self.target_net = DQNNetwork(data.num_items, self.n_actions, cfg).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.replay_capacity)
        self.step_count = 0

    def select_actions(self, obs, eps, epsilon: float) -> np.ndarray:
        mask = _mask_remaining(eps, self.n_actions)
        with torch.no_grad():
            q = self.net(obs)
        q = q.cpu().numpy() * mask
        q[mask == 0] = -1e9
        greedy = q.argmax(-1)
        explore = np.array([np.random.choice(np.where(mask[i] > 0)[0])
                            for i in range(len(eps))])
        use_explore = np.random.rand(len(eps)) < epsilon
        return np.where(use_explore, explore, greedy)

    def collect(self, users: np.ndarray) -> Dict[str, float]:
        eps = self.env.reset_batch(users)
        dev = self.device
        returns, totals = [], []
        for _ in range(self.cfg.interaction_length):
            obs = self.env.build_obs(eps, dev)
            obs_np = [t.cpu().numpy() for t in obs]
            actions = self.select_actions(obs, eps, epsilon=self.cfg.epsilon)
            rewards, dones = [], []
            for ep, a in zip(eps, actions):
                rew = self.env.step(ep, int(a))
                rewards.append(rew["total"])
                dones.append(self.env.done(ep))
            next_obs = self.env.build_obs(eps, dev)
            _push(self.buffer, obs_np, actions, rewards,
                  [t.cpu().numpy() for t in next_obs], dones)
            returns.append(float(np.sum(rewards)))
            totals.append(float(np.mean(rewards)))
            if all(dones):
                break
        return {"episode_return": float(np.mean(returns)),
                "mean_step_reward": float(np.mean(totals))}

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {}
        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        obs = (batch["pos"], batch["neg"], batch["seq"])
        obs2 = (batch["next_pos"], batch["next_neg"], batch["next_seq"])
        a, r, done = batch["action"], batch["reward"], batch["done"]

        q = self.net(obs).gather(1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            if self.cfg.double_q:                       # DDQN
                a_next = self.net(obs2).argmax(-1, keepdim=True)
                q_next = self.target_net(obs2).gather(1, a_next).squeeze(-1)
            else:                                       # DQN-R
                q_next = self.target_net(obs2).max(-1).values
            y = r + self.cfg.gamma * (1 - done) * q_next
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.step_count += 1
        if self.step_count % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        return {"dqn_loss": float(loss.item())}

    def evaluate(self, users: np.ndarray) -> List[EpisodeResult]:
        self.net.eval()
        eps = self.env.reset_batch(users)
        dev = self.device
        results = [EpisodeResult(user=int(u)) for u in users]
        for _ in range(self.cfg.interaction_length):
            obs = self.env.build_obs(eps, dev)
            actions = self.select_actions(obs, eps, epsilon=0.0)
            for i, (ep, a) in enumerate(zip(eps, actions)):
                rew = self.env.step(ep, int(a))
                results[i].seq.append(int(a))
                results[i].racc.append(rew["racc"])
                results[i].total.append(rew["total"])
            if all(self.env.done(ep) for ep in eps):
                break
        self.net.train()
        return results

    # -- warm-start checkpoints --------------------------------------------
    def state_dict(self, include_buffer: bool = False) -> dict:
        d = {
            "agent": self.name,
            "net": self.net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "opt": self.opt.state_dict(),
            "step_count": self.step_count,
            "buffer": None,
        }
        if include_buffer:
            d["buffer"] = list(self.buffer.buf)
        return d

    def load_state_dict(self, d: dict) -> None:
        if d.get("agent") and d["agent"] != self.name:
            print(f"[warn] checkpoint agent '{d['agent']}' does not match "
                  f"'{self.name}'; loading anyway")
        self.net.load_state_dict(d["net"])
        self.target_net.load_state_dict(d["target_net"])
        self.opt.load_state_dict(d["opt"])
        self.step_count = d.get("step_count", 0)
        buf = d.get("buffer")
        if buf:
            self.buffer.buf = deque(buf, maxlen=self.cfg.replay_capacity)


# ---------------------------------------------------------------------------
# Rainbow agent (dueling + distributional + noisy + PER + n-step + DDQN)
# ---------------------------------------------------------------------------
class RainbowAgent:
    name = "Rainbow"

    def __init__(self, cfg, data, env):
        self.cfg = cfg
        self.data = data
        self.env = env
        self.device = torch.device(cfg.device)
        self.n_actions = data.num_items
        self.n_atoms = cfg.n_atoms
        self.v_min, self.v_max = cfg.v_min, cfg.v_max
        self.support = torch.linspace(cfg.v_min, cfg.v_max, cfg.n_atoms,
                                      device=self.device)
        self.net = RainbowNetwork(data.num_items, self.n_actions, cfg).to(self.device)
        self.target_net = RainbowNetwork(data.num_items, self.n_actions, cfg).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buffer = PrioritizedReplayBuffer(cfg.replay_capacity, cfg.per_alpha)
        self.step_count = 0
        self.beta = cfg.per_beta0

    # -- action selection (greedy over expected Q; noisy nets explore) -----
    def select_actions(self, obs, eps) -> np.ndarray:
        mask = _mask_remaining(eps, self.n_actions)
        with torch.no_grad():
            q = self.net.expected_values(obs, self.support)
        q = q.cpu().numpy() * mask
        q[mask == 0] = -1e9
        return q.argmax(-1)

    # -- n-step episode collection -----------------------------------------
    def collect(self, users: np.ndarray) -> Dict[str, float]:
        eps = self.env.reset_batch(users)
        dev = self.device
        n = self.cfg.multi_step
        windows: List[deque] = [deque(maxlen=n) for _ in eps]
        returns, totals = [], []
        for t in range(self.cfg.interaction_length):
            obs = self.env.build_obs(eps, dev)
            obs_np = [t_.cpu().numpy() for t_ in obs]
            actions = self.select_actions(obs, eps)
            rewards, dones = [], []
            for i, (ep, a) in enumerate(zip(eps, actions)):
                rew = self.env.step(ep, int(a))
                rewards.append(rew["total"])
                dones.append(self.env.done(ep))
            next_obs = self.env.build_obs(eps, dev)
            next_obs_np = [t_.cpu().numpy() for t_ in next_obs]

            for i in range(len(eps)):
                win = windows[i]
                win.append((obs_np[0][i], obs_np[1][i], obs_np[2][i],
                            actions[i], rewards[i]))
                if dones[i]:
                    # flush pending entries with truncated returns
                    pend = list(win)
                    for j, (p0, p1, p2, aa, rr) in enumerate(pend):
                        ret = sum(self.cfg.gamma ** k * pend[j + k][4]
                                  for k in range(len(pend) - j))
                        self.buffer.push((p0, p1, p2, aa, ret,
                                          next_obs_np[0][i], next_obs_np[1][i],
                                          next_obs_np[2][i], True))
                    win.clear()
                elif len(win) == n:
                    p0, p1, p2, aa, rr = win.popleft()
                    ret = rr + sum(self.cfg.gamma ** (k + 1) * win[k][4]
                                   for k in range(len(win)))
                    self.buffer.push((p0, p1, p2, aa, ret,
                                      next_obs_np[0][i], next_obs_np[1][i],
                                      next_obs_np[2][i], False))
            returns.append(float(np.sum(rewards)))
            totals.append(float(np.mean(rewards)))
            if all(dones):
                break
        return {"episode_return": float(np.mean(returns)),
                "mean_step_reward": float(np.mean(totals))}

    # -- distributional (C51) update ---------------------------------------
    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {}
        self.beta = min(1.0, self.cfg.per_beta0
                        + self.step_count * (1.0 - self.cfg.per_beta0)
                        / max(1, self.cfg.per_beta_steps))
        batch, weights, idx = self.buffer.sample(self.cfg.batch_size, self.beta,
                                                 self.device)
        obs = (batch["pos"], batch["neg"], batch["seq"])
        obs2 = (batch["next_pos"], batch["next_neg"], batch["next_seq"])
        a, r, done = batch["action"], batch["reward"], batch["done"]
        B = len(a)

        # online distribution of selected actions
        logits = self.net(obs)                                  # (B, at, |A|)
        p_sel = F.softmax(logits, dim=1).gather(
            2, a.view(B, 1, 1).expand(B, self.n_atoms, 1)).squeeze(-1)  # (B, at)

        # target distribution (double Q over expected values)
        with torch.no_grad():
            nxt = self.target_net.probs(obs2)                   # (B, at, |A|)
            q_next = (nxt * self.support.view(1, -1, 1)).sum(1)  # (B, |A|)
            a_next = q_next.argmax(-1)
            p_next = nxt.gather(
                2, a_next.view(B, 1, 1).expand(B, self.n_atoms, 1)).squeeze(-1)
            z = self.support.view(1, -1)
            t_z = (r.unsqueeze(-1)
                   + (1 - done.unsqueeze(-1))
                   * self.cfg.gamma ** self.cfg.multi_step * z)
            t_z = t_z.clamp(self.v_min, self.v_max)
            b_ = (t_z - self.v_min) / (self.v_max - self.v_min) * (self.n_atoms - 1)
            lo = b_.floor().long().clamp(0, self.n_atoms - 1)
            hi = b_.ceil().long().clamp(0, self.n_atoms - 1)
            frac = (b_ - b_.floor()).clamp(0, 1)

            m = torch.zeros(B, self.n_atoms, self.n_atoms, device=self.device)
            for j in range(self.n_atoms):
                m[:, j, lo[:, j]] += p_next[:, j] * (1 - frac[:, j])
                m[:, j, hi[:, j]] += p_next[:, j] * frac[:, j]
            target_dist = m.sum(1)                              # (B, at)

        loss = -(target_dist * p_sel.clamp_min(1e-8).log()).sum(1)
        loss = (torch.as_tensor(weights, device=self.device) * loss).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        td = (target_dist - p_sel).abs().sum(1).detach().cpu().numpy()
        self.buffer.update_priorities(idx, td)

        self.step_count += 1
        if self.step_count % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        return {"rainbow_loss": float(loss.item())}

    def evaluate(self, users: np.ndarray) -> List[EpisodeResult]:
        self.net.eval()
        eps = self.env.reset_batch(users)
        dev = self.device
        results = [EpisodeResult(user=int(u)) for u in users]
        for _ in range(self.cfg.interaction_length):
            obs = self.env.build_obs(eps, dev)
            actions = self.select_actions(obs, eps)
            for i, (ep, a) in enumerate(zip(eps, actions)):
                rew = self.env.step(ep, int(a))
                results[i].seq.append(int(a))
                results[i].racc.append(rew["racc"])
                results[i].total.append(rew["total"])
            if all(self.env.done(ep) for ep in eps):
                break
        self.net.train()
        return results

    # -- warm-start checkpoints --------------------------------------------
    def state_dict(self, include_buffer: bool = False) -> dict:
        d = {
            "agent": self.name,
            "net": self.net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "opt": self.opt.state_dict(),
            "step_count": self.step_count,
            "beta": self.beta,
            "buffer": None,
        }
        if include_buffer:
            b = self.buffer
            d["buffer"] = {"buf": b.buf, "priorities": b.priorities,
                            "pos": b.pos, "size": b.size, "alpha": b.alpha}
        return d

    def load_state_dict(self, d: dict) -> None:
        if d.get("agent") and d["agent"] != self.name:
            print(f"[warn] checkpoint agent '{d['agent']}' does not match "
                  f"'{self.name}'; loading anyway")
        self.net.load_state_dict(d["net"])
        self.target_net.load_state_dict(d["target_net"])
        self.opt.load_state_dict(d["opt"])
        self.step_count = d.get("step_count", 0)
        self.beta = d.get("beta", self.cfg.per_beta0)
        b = d.get("buffer")
        if b:
            self.buffer.buf = b["buf"]
            self.buffer.priorities = b["priorities"]
            self.buffer.pos = b["pos"]
            self.buffer.size = b["size"]
            self.buffer.alpha = b.get("alpha", self.cfg.per_alpha)


AGENT_CLASSES = {
    "sacd": SACDAgent,
    "dqn": DQNAgent,
    "rainbow": RainbowAgent,
}


def build_agent(cfg, data, env):
    return AGENT_CLASSES[cfg.agent](cfg, data, env)
