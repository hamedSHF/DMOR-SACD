"""Experience replay buffers.

* ReplayBuffer             -- uniform sampling (SACD, DQN-R, DDQN).
* PrioritizedReplayBuffer  -- proportional prioritisation + importance
                              sampling weights (Rainbow baseline).

Transitions are stored as (pos, neg, seq, action, reward, next_pos,
next_neg, next_seq, done) -- the three ID tensors are the raw observations.
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, transition: tuple) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int, device) -> dict:
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        return self._collate([self.buf[i] for i in idx], device)

    @staticmethod
    def _collate(batch: List[tuple], device) -> dict:
        pos = np.stack([t[0] for t in batch])
        neg = np.stack([t[1] for t in batch])
        seq = np.stack([t[2] for t in batch])
        act = np.array([t[3] for t in batch], dtype=np.int64)
        rew = np.array([t[4] for t in batch], dtype=np.float32)
        npos = np.stack([t[5] for t in batch])
        nneg = np.stack([t[6] for t in batch])
        nseq = np.stack([t[7] for t in batch])
        done = np.array([t[8] for t in batch], dtype=np.float32)
        return {
            "pos": torch.as_tensor(pos, device=device),
            "neg": torch.as_tensor(neg, device=device),
            "seq": torch.as_tensor(seq, device=device),
            "action": torch.as_tensor(act, device=device),
            "reward": torch.as_tensor(rew, device=device),
            "next_pos": torch.as_tensor(npos, device=device),
            "next_neg": torch.as_tensor(nneg, device=device),
            "next_seq": torch.as_tensor(nseq, device=device),
            "done": torch.as_tensor(done, device=device),
        }


class PrioritizedReplayBuffer:
    """Proportional prioritisation (Schaul et al. 2016)."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buf: List[tuple] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push(self, transition: tuple, priority: float = 1.0) -> None:
        if self.size < self.capacity:
            self.buf.append(transition)
        else:
            self.buf[self.pos] = transition
        self.priorities[self.pos] = max(priority, 1e-6)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float, device) -> Tuple[dict, np.ndarray, np.ndarray]:
        p = self.priorities[:self.size] ** self.alpha
        probs = p / p.sum()
        idx = np.random.choice(self.size, size=batch_size, p=probs)
        weights = (self.size * probs[idx]) ** (-beta)
        weights /= weights.max()
        batch = [self.buf[i] for i in idx]
        return self._collate(batch, device), weights.astype(np.float32), idx

    @staticmethod
    def _collate(batch: List[tuple], device) -> dict:
        return ReplayBuffer._collate(batch, device)

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        for i, e in zip(idx, td_errors):
            self.priorities[i] = max(float(abs(e)), 1e-6)
