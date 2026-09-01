"""Central configuration for DMoR-SACD.

Hyper-parameters follow the paper (Section 4.1, "Parameter settings"):
    * sequence / history length  N = 10
    * interaction length         K = 20
    * embedding & hidden size    50
    * discount factor            gamma = 0.9
    * learning rate              2.5e-4 (actor, critic, alpha)
    * batch size                 256
    * replay buffer capacity     1e5
    * soft update rate           tau = 0.01
    * target entropy rate        0.8  ->  H_bar = -0.8 * log(1/|A|)
    * satisfaction/diversity coefs lambda1 = lambda2 = 1/|A|
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Reward modes -------------------------------------------------------------
ACCURACY = "accuracy"                 # racc            (all methods)
SATISFACTION = "satisfaction"         # racc + rsat     (Model 3-sat)
DIVERSITY = "diversity"               # racc + rdiv     (Model 3-div)
MULTI = "multi"                       # racc+rsat+rdiv  (DMoR-SACD / Model 2)

# State encoders -----------------------------------------------------------
GRU_STATE = "gru"                     # two-GRU state component (Eq. 3)
POOL_STATE = "pool"                   # simple mean-pooled item-sequence state

# Agents -------------------------------------------------------------------
SACD = "sacd"
DQN = "dqn"
RAINBOW = "rainbow"

# Datasets -----------------------------------------------------------------
ML_100K = "ml-100k"
ML_1M = "ml-1m"
LASTFM = "lastfm"
RETAILROCKET = "retailrocket"


@dataclass
class Config:
    # --- data -------------------------------------------------------------
    dataset: str = ML_100K
    data_dir: str = "data"
    seed: int = 42
    # users with fewer than this many interactions are dropped (paper: >= 20)
    min_interactions: int = 20
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    positive_threshold: float = 3.0   # rating >= 3  -> positive item
    # RetailRocket only: keep the top-K most frequent items as the candidate
    # set (the full dataset has ~235K items; 0 = keep every item)
    max_items: int = 10_000

    # --- MDP / environment ------------------------------------------------
    history_len: int = 10             # N : length of s+_t and s-_t queues
    window_size: int = 10             # W : sliding window for diversity
    interaction_length: int = 20      # K : episode length
    no_repeat: bool = True            # an item is never recommended twice
                                      #     within one episode

    # --- network ----------------------------------------------------------
    embedding_dim: int = 50
    hidden_dim: int = 50

    # --- RL ---------------------------------------------------------------
    gamma: float = 0.9
    lr: float = 2.5e-4
    alpha_lr: float = 2.5e-4
    batch_users: int = 256            # users sampled per episode
    batch_size: int = 256             # gradient mini-batch
    replay_capacity: int = 100_000
    tau: float = 0.01                 # target soft-update
    target_entropy_rate: float = 0.8  # H_bar = -rate * log(1/|A|)
    episodes: int = 20_000
    updates_per_episode: int = 1
    eval_every: int = 2_000
    eval_users: int = 100
    log_every: int = 500

    # --- DQN family -------------------------------------------------------
    epsilon: float = 0.1              # DQN-R epsilon-greedy (paper)
    double_q: bool = False            # DDQN
    target_update_freq: int = 200     # hard copy frequency for DQN family

    # --- Rainbow ----------------------------------------------------------
    n_atoms: int = 51
    v_min: float = -1.0
    v_max: float = 1.0
    multi_step: int = 3
    per_alpha: float = 0.6
    per_beta0: float = 0.4
    per_beta_steps: int = 20_000
    noisy_sigma0: float = 0.5

    # --- training runtime -------------------------------------------------
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    num_workers: int = 0
    smoke: bool = False               # tiny run for sanity checking
    seed_eval_users: bool = True      # deterministic eval user set
    save_dir: str = "runs"           # where weights/logs are written

    # model identity (filled by presets) -----------------------------------
    agent: str = SACD
    reward_mode: str = MULTI
    state_encoder: str = GRU_STATE

    # ------------------------------------------------------------------
    def resolve(self) -> "Config":
        if self.smoke:
            self.episodes = 60
            self.batch_users = 128
            self.batch_size = 64
            self.eval_users = 50
            self.eval_every = 20
            self.log_every = 10
            self.replay_capacity = 5_000
            self.per_beta_steps = 1_000
            self.target_update_freq = 20
        return self


MODEL_PRESETS = {
    # paper name -> (agent, reward_mode, state_encoder)
    "dmorsacd":   (SACD, MULTI, GRU_STATE),
    "model1":     (SACD, ACCURACY, POOL_STATE),      # SACD alone
    "model2":     (SACD, MULTI, POOL_STATE),         # SACD + reward component
    "model3":     (SACD, ACCURACY, GRU_STATE),       # SACD + GRUs
    "model3-sat": (SACD, SATISFACTION, GRU_STATE),   # + satisfaction goal
    "model3-div": (SACD, DIVERSITY, GRU_STATE),      # + diversity goal
    "dqn-r":      (DQN, ACCURACY, GRU_STATE),        # DQN with two-GRU state
    "ddqn":       (DQN, ACCURACY, POOL_STATE),       # double DQN (plain state)
    "rainbow":    (RAINBOW, ACCURACY, POOL_STATE),   # Rainbow (plain state)
}

ALL_MODELS = list(MODEL_PRESETS.keys())


def preset_config(model: str, **overrides) -> Config:
    """Return a Config for one of the paper's models/baselines."""
    if model not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model '{model}'. Choose from {ALL_MODELS}."
        )
    agent, reward_mode, state_encoder = MODEL_PRESETS[model]
    cfg = Config(agent=agent, reward_mode=reward_mode, state_encoder=state_encoder)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown config field '{k}'")
        setattr(cfg, k, v)
    cfg.resolve()
    return cfg
