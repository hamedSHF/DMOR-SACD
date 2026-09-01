"""Offline recommendation environment (Section 3.1 & 3.2).

The recommender agent interacts with logged user feedback:

* state  s_t = {s+_t, s-_t}: queues holding the last N = 10 positively /
  negatively received items of user u before time step t (Eq. 2).
  On positive feedback  a_t is appended to s+_t (oldest dropped); on
  negative (skip) feedback it is appended to s-_t.
* interest code  v_i: +1 for a positive item, -1 for a negative item,
  0 for steps not yet reached -- built from the equivalence transformation.
* item sequence: all items recommended so far (both signs), which together
  with the interest code feeds the reward component.
* sliding window W (length N = 10): the last N recommended items, used by
  the diversity reward.

Items are never recommended twice within one episode (paper, Section 3.5).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class EpisodeState:
    user: int
    pos: deque = field(default_factory=lambda: deque(maxlen=10))
    neg: deque = field(default_factory=lambda: deque(maxlen=10))
    seq: List[int] = field(default_factory=list)
    interest: List[int] = field(default_factory=list)
    window: deque = field(default_factory=lambda: deque(maxlen=10))
    prev_fdiv: float = 0.0


class RecommendationEnv:
    """Simulates K-step recommendation episodes from offline logs."""

    def __init__(self, cfg, data, reward):
        self.cfg = cfg
        self.data = data
        self.reward = reward
        self.n = cfg.history_len
        self.k = cfg.interaction_length
        self.no_repeat = cfg.no_repeat
        self.n_items = data.num_items

    # -- lifecycle ---------------------------------------------------------
    def reset(self, user: int) -> EpisodeState:
        return EpisodeState(user=user,
                            pos=deque(maxlen=self.n),
                            neg=deque(maxlen=self.n),
                            window=deque(maxlen=self.n))

    def reset_batch(self, users: np.ndarray) -> List[EpisodeState]:
        return [self.reset(int(u)) for u in users]

    # -- one step ----------------------------------------------------------
    def step(self, ep: EpisodeState, item: int) -> Dict[str, float]:
        """Recommend `item` to the user; update state; return rewards."""
        r = self.data.rating(ep.user, item)
        positive = r > 0 and r >= self.cfg.positive_threshold
        window_full = len(ep.window) == self.n
        window_before = list(ep.window)

        rew = self.reward(
            ep.user, item, list(ep.interest), window_before,
            ep.prev_fdiv, window_full,
        )

        # --- state transition (Section 3.2.1) -----------------------------
        ep.seq.append(item)
        ep.interest.append(1 if positive else -1)
        ep.window.append(item)               # at enters W, oldest dropped
        ep.prev_fdiv = rew["fdiv"]
        if positive:
            ep.pos.append(item)              # s'+ = {a+_2..a+_10, a_t}
        else:
            ep.neg.append(item)              # user skips -> negative queue
        return rew

    def done(self, ep: EpisodeState) -> bool:
        if len(ep.seq) >= self.k:
            return True
        if self.no_repeat and len(ep.seq) >= self.n_items:
            return True
        return False

    # -- observations ------------------------------------------------------
    def build_obs(self, eps: List[EpisodeState], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded ID tensors (B, N) for (positive, negative, sequence)."""
        pos = np.zeros((len(eps), self.n), dtype=np.int64)
        neg = np.zeros((len(eps), self.n), dtype=np.int64)
        seq = np.zeros((len(eps), self.n), dtype=np.int64)
        for i, ep in enumerate(eps):
            p = list(ep.pos)
            ng = list(ep.neg)
            s = list(ep.seq)[-self.n:]
            pos[i, self.n - len(p):] = p
            neg[i, self.n - len(ng):] = ng
            seq[i, self.n - len(s):] = s
        return (torch.as_tensor(pos, device=device),
                torch.as_tensor(neg, device=device),
                torch.as_tensor(seq, device=device))
