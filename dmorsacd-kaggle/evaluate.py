"""Offline evaluation metrics (Section 4.1).

    R@K  average cumulative reward of the recommended sequence   (Eq. 17)
    H@K  hit rate: fraction of recommended items that are in the
              user's interaction history                              (Eq. 18)
    N@K  normalised discounted cumulative gain                   (Eq. 19)
    D@K  average pairwise dissimilarity of the sequence          (Eq. 20)

Testing follows Algorithm 2: the item with maximum probability / Q-value
is recommended at each step and removed from the available set.
"""
from __future__ import annotations

import numpy as np

from .agents import EpisodeResult


def compute_metrics(results, data, k: int) -> dict:
    """Compute R@K, H@K, N@K, D@K over a list of EpisodeResults."""
    R, H, N, D = [], [], [], []
    for res in results:
        seq = res.seq[:k]
        if not seq:
            continue
        # R@K: sum of normalised ratings (Eq. 17)
        R.append(float(np.sum(res.racc[:k])))

        # H@K / N@K: relevance = item appears in the user's history (Eq. 18)
        y = np.array([1.0 if data.rating(res.user, it) > 0 else 0.0
                      for it in seq], dtype=np.float64)
        H.append(float(y.mean()))
        discounts = np.log2(np.arange(len(y)) + 2.0)
        dcg = float((y / discounts).sum())
        idcg = float((1.0 / discounts).sum())
        N.append(dcg / idcg if idcg > 0 else 0.0)

        # D@K: average pairwise dissimilarity (Eq. 20)
        n = len(seq)
        if n > 1:
            sims = np.array([1.0 - data.item_sim(seq[i], seq[j])
                             for i in range(n) for j in range(i + 1, n)])
            D.append(float(2.0 * sims.sum() / (n * (n - 1))))
        else:
            D.append(0.0)

    return {
        "R@K": float(np.mean(R)),
        "H@K": float(np.mean(H)),
        "N@K": float(np.mean(N)),
        "D@K": float(np.mean(D)),
    }


def evaluate_agent(agent, data, users, cfg) -> dict:
    """Run deterministic episodes for `users` and return the metrics."""
    results = agent.evaluate(np.asarray(users, dtype=np.int64))
    return compute_metrics(results, data, cfg.interaction_length)
