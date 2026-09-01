"""Neural networks for DMoR-SACD and its baselines.

State component (Section 3.2.1, Eq. 3):
    p_t = GRU(s+_t),  n_t = GRU(s-_t)
    p'_t = MLP(p_t),  n'_t = MLP(n_t)
    s'_t = MLP(concat(p'_t, n'_t))

SACD policy component (Section 3.4, Fig. 3):
    actor    : state -> softmax(|A|)     (categorical policy pi_theta)
    critics  : state -> Q-values for every action (|A|)
    targets  : two target Q-networks, min(Q'_1, Q'_2)

Baselines:
    * DQN-R   : two-GRU state component + DQN head (Zhao et al. 2018)
    * DDQN    : plain pooled state + double DQN head (van Hasselt et al. 2016)
    * Rainbow : pooled state + dueling + distributional + noisy nets
                (Hessel et al. 2018)

Weights are He-initialised (paper, Section 4.1); the actor's last layer uses
Softmax, all other layers ReLU.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GRU_STATE, POOL_STATE


def he_init(m: nn.Module) -> nn.Module:
    """He initialiser (kaiming uniform) applied to Linear/GRU weights."""
    for name, p in m.named_parameters():
        if p.ndim >= 2:
            nn.init.kaiming_uniform_(p, a=5 ** 0.5)
        else:
            try:
                nn.init.uniform_(p, -0.05, 0.05)
            except ValueError:
                pass
    return m


# ---------------------------------------------------------------------------
# State encoders
# ---------------------------------------------------------------------------
class GRUStateEncoder(nn.Module):
    """Two-GRU state component (Eq. 3) with 0-padding and final hidden state."""

    def __init__(self, num_items, cfg):
        super().__init__()
        d = cfg.embedding_dim
        h = cfg.hidden_dim
        self.embedding = nn.Embedding(num_items + 1, d, padding_idx=0)
        self.gru_pos = nn.GRU(d, h, batch_first=True)
        self.gru_neg = nn.GRU(d, h, batch_first=True)
        self.mlp_pos = nn.Linear(h, h)
        self.mlp_neg = nn.Linear(h, h)
        self.mlp_mix = nn.Linear(2 * h, h)
        he_init(self)

    def forward(self, pos_ids: torch.Tensor, neg_ids: torch.Tensor) -> torch.Tensor:
        # (B, N) -> (B, h)
        p = self.embedding(pos_ids)
        n = self.embedding(neg_ids)
        _, p_h = self.gru_pos(p)            # final hidden states
        _, n_h = self.gru_neg(n)
        p_h, n_h = p_h.squeeze(0), n_h.squeeze(0)
        p_p = F.relu(self.mlp_pos(p_h))
        n_p = F.relu(self.mlp_neg(n_h))
        mix = torch.cat([p_p, n_p], dim=-1)
        return F.relu(self.mlp_mix(mix))


class PoolStateEncoder(nn.Module):
    """Mean-pooled item-sequence state (used by the no-GRU variants)."""

    def __init__(self, num_items, cfg):
        super().__init__()
        d = cfg.embedding_dim
        h = cfg.hidden_dim
        self.embedding = nn.Embedding(num_items + 1, d, padding_idx=0)
        self.mlp = nn.Linear(d, h)
        he_init(self)

    def forward(self, seq_ids: torch.Tensor, **_) -> torch.Tensor:
        e = self.embedding(seq_ids)                       # (B, N, d)
        mask = (seq_ids != 0).unsqueeze(-1).float()
        pooled = (e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.relu(self.mlp(pooled))


def make_encoder(encoder_type: str, num_items, cfg) -> nn.Module:
    """Build the state encoder selected by cfg."""
    if encoder_type == GRU_STATE:
        return GRUStateEncoder(num_items, cfg)
    if encoder_type == POOL_STATE:
        return PoolStateEncoder(num_items, cfg)
    raise ValueError(f"Unknown state encoder '{encoder_type}'")


# ---------------------------------------------------------------------------
# SACD networks (Section 3.4.1)
# ---------------------------------------------------------------------------
class ActorHead(nn.Module):
    """pi_theta : s' -> softmax over all |A| items."""

    def __init__(self, cfg, n_actions: int):
        super().__init__()
        h = cfg.hidden_dim
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, n_actions)
        he_init(self)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(feat))
        return F.softmax(self.fc2(x), dim=-1)


class QHead(nn.Module):
    """Q_phi : s' -> soft Q-value of every item (|A|)."""

    def __init__(self, cfg, n_actions: int):
        super().__init__()
        h = cfg.hidden_dim
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, n_actions)
        he_init(self)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(feat))
        return self.fc2(x)


class SACDNetwork(nn.Module):
    """Actor + two critics + two target critics (Fig. 3)."""

    def __init__(self, num_items: int, n_actions: int, cfg):
        super().__init__()
        self.num_items = num_items
        self.n_actions = n_actions
        self.encoder_type = cfg.state_encoder

        self.actor_encoder = make_encoder(cfg.state_encoder, num_items, cfg)
        self.actor = ActorHead(cfg, n_actions)
        self.critic1 = nn.ModuleList([make_encoder(cfg.state_encoder, num_items, cfg),
                                      QHead(cfg, n_actions)])
        self.critic2 = nn.ModuleList([make_encoder(cfg.state_encoder, num_items, cfg),
                                      QHead(cfg, n_actions)])
        # target networks: full copies (shared structure, different weights)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

    # -- encoders ----------------------------------------------------------
    def _feat(self, enc, obs) -> torch.Tensor:
        pos_ids, neg_ids, seq_ids = obs
        if isinstance(enc, GRUStateEncoder):
            return enc(pos_ids, neg_ids)
        return enc(seq_ids)

    def actor_feature(self, obs) -> torch.Tensor:
        return self._feat(self.actor_encoder, obs)

    # -- actor -------------------------------------------------------------
    def actor_probs(self, obs) -> torch.Tensor:
        return self.actor(self.actor_feature(obs))

    # -- critics -----------------------------------------------------------
    def critic_values(self, idx: int, obs) -> torch.Tensor:
        enc, head = (self.critic1 if idx == 1 else self.critic2)
        return head(self._feat(enc, obs))

    def min_critic(self, obs) -> torch.Tensor:
        """min(Q_phi1, Q_phi2) used by the actor loss (Eq. 15)."""
        return torch.minimum(self.critic_values(1, obs),
                             self.critic_values(2, obs))

    def target_values(self, obs) -> torch.Tensor:
        """min(Q'_phi1, Q'_phi2) used by the critic target (Eq. 13)."""
        e1, h1 = self.target_critic1
        e2, h2 = self.target_critic2
        q1 = h1(self._feat(e1, obs))
        q2 = h2(self._feat(e2, obs))
        return torch.minimum(q1, q2)

    def soft_update(self, tau: float) -> None:
        for target, source in (
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2),
        ):
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ---------------------------------------------------------------------------
# DQN family networks
# ---------------------------------------------------------------------------
class DQNNetwork(nn.Module):
    """Q-network with the selected state encoder (DQN-R: GRU, DDQN: pool)."""

    def __init__(self, num_items: int, n_actions: int, cfg):
        super().__init__()
        self.encoder = make_encoder(cfg.state_encoder, num_items, cfg)
        h = cfg.hidden_dim
        self.q_head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions),
        )
        he_init(self)

    def forward(self, obs) -> torch.Tensor:
        pos_ids, neg_ids, seq_ids = obs
        if isinstance(self.encoder, GRUStateEncoder):
            feat = self.encoder(pos_ids, neg_ids)
        else:
            feat = self.encoder(seq_ids)
        return self.q_head(feat)


class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy layer (Fortunato et al. 2018)."""

    def __init__(self, in_f: int, out_f: int, sigma0: float = 0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        mu_range = 1.0 / in_f ** 0.5
        self.w_mu = nn.Parameter(torch.empty(out_f, in_f).uniform_(-mu_range, mu_range))
        self.b_mu = nn.Parameter(torch.empty(out_f).uniform_(-mu_range, mu_range))
        self.w_sigma = nn.Parameter(torch.full((out_f, in_f), sigma0 / in_f ** 0.5))
        self.b_sigma = nn.Parameter(torch.full((out_f,), sigma0 / in_f ** 0.5))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def _sample_eps(self, size: int, device) -> torch.Tensor:
        return torch.randn(size, device=device).sign_().mul_(
            torch.randn(size, device=device).abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            e_in = self._sample_eps(x.shape[-1], x.device)
            e_out = self._sample_eps(self.out_f, x.device)
            w = self.w_mu + self.w_sigma * self._f(e_out).outer(self._f(e_in))
            b = self.b_mu + self.b_sigma * self._f(e_out)
        else:
            w, b = self.w_mu, self.b_mu
        return F.linear(x, w, b)


class RainbowNetwork(nn.Module):
    """Dueling + distributional (C51) + noisy-nets Rainbow Q-network."""

    def __init__(self, num_items: int, n_actions: int, cfg):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = cfg.n_atoms
        self.encoder = PoolStateEncoder(num_items, cfg)
        h = cfg.hidden_dim
        self.value = nn.Sequential(NoisyLinear(h, h, cfg.noisy_sigma0),
                                   nn.ReLU(),
                                   NoisyLinear(h, self.n_atoms, cfg.noisy_sigma0))
        self.advantage = nn.Sequential(NoisyLinear(h, h, cfg.noisy_sigma0),
                                       nn.ReLU(),
                                       NoisyLinear(h, self.n_actions * self.n_atoms,
                                                   cfg.noisy_sigma0))

    def forward(self, obs) -> torch.Tensor:
        """Return logits of shape (B, n_atoms, |A|)."""
        pos_ids, neg_ids, seq_ids = obs
        feat = self.encoder(seq_ids)
        v = self.value(feat).view(-1, self.n_atoms, 1)          # (B, at, 1)
        a = self.advantage(feat).view(-1, self.n_actions, self.n_atoms)
        a = a - a.mean(dim=1, keepdim=True)
        logits = v + a.transpose(1, 2)             # (B, atoms, |A|)
        return logits

    def probs(self, obs) -> torch.Tensor:
        return F.softmax(self.forward(obs), dim=1)

    def expected_values(self, obs, support: torch.Tensor) -> torch.Tensor:
        p = self.probs(obs)                        # (B, atoms, |A|)
        return (p * support.view(1, -1, 1)).sum(dim=1)
