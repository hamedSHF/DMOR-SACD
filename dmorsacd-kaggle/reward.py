"""Reward component (Section 3.3).

    R(s_t, a_t) = racc + lambda1 * rsat + lambda2 * rdiv          (Eq. 4)

* racc -- accuracy reward: linear normalisation of the raw rating to [-1, 1]
          (Eq. 5); items without an interaction record get -1 (Fu et al. 2022).
* rsat -- satisfaction reward built on the interest code v_i (Eqs. 6-7).
* rdiv -- diversity reward built on a sliding window of the last N
          recommended items with cosine similarity (Eqs. 8-9); a switch keeps
          it off until the window is full.

Paper: lambda1 = lambda2 = 1/|A|.
"""
from __future__ import annotations

from typing import Optional

from .config import ACCURACY, DIVERSITY, MULTI, SATISFACTION


def satisfaction_score(interest: list) -> float:
    """Eq. 6: fsat(v_i) = p - n + p_succ - n_succ.

    p/n:  number of positive/negative items recommended so far.
    p_succ/n_succ: for every run of >=3 consecutive +1 (or -1) in the
    interest code, each occurrence beyond the first two adds +1.
    """
    p = interest.count(1)
    n = interest.count(-1)
    p_succ = n_succ = 0
    run_sign, run_len = 0, 0
    for v in interest:
        if v == run_sign:
            run_len += 1
        else:
            run_sign, run_len = v, 1
        if run_len > 2:
            if run_sign == 1:
                p_succ += 1
            elif run_sign == -1:
                n_succ += 1
    return float(p - n + p_succ - n_succ)


class RewardComponent:
    """Computes the (dynamic) multi-objective reward for one step."""

    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.mode = cfg.reward_mode
        self.k = float(cfg.interaction_length)
        self.lambda1 = 1.0 / data.num_items
        self.lambda2 = 1.0 / data.num_items
        self.use_sat = self.mode in (MULTI, SATISFACTION)
        self.use_div = self.mode in (MULTI, DIVERSITY)

    # -- accuracy ----------------------------------------------------------
    def accuracy(self, user: int, item: int) -> float:
        r = self.data.rating(user, item)
        if r <= 0.0:
            return -1.0                       # no interaction record -> -1
        return self.data.normalize_rating(r)  # Eq. 5

    # -- satisfaction ------------------------------------------------------
    def satisfaction(self, interest: list) -> float:
        """Eq. 7: rsat = fsat(v_i) / K."""
        return satisfaction_score(interest) / self.k

    # -- diversity ---------------------------------------------------------
    @staticmethod
    def fdiv(data, item: int, window: list) -> float:
        """Eq. 8: fdiv(a_t) = 1/N * sum_{a_i in W} (1 - sim(a_t, a_i))."""
        if not window:
            return 0.0
        n = len(window)
        return (1.0 / n) * sum(1.0 - data.item_sim(item, a) for a in window)

    # -- composite ---------------------------------------------------------
    def __call__(self, user: int, item: int, interest: list,
                 window: list, prev_fdiv: float, window_full: bool) -> dict:
        """Return {racc, rsat, rdiv, total} for recommending `item`.

        `prev_fdiv` is fdiv(a_{t-1}) -- 0.0 while the diversity switch is off,
        so rdiv = fdiv(a_t) - fdiv(a_{t-1})  (Eq. 9).
        """
        racc = self.accuracy(user, item)
        rsat = self.satisfaction(interest) if self.use_sat else 0.0
        if self.use_div and window_full:
            fdiv_t = self.fdiv(self.data, item, window)
            rdiv = fdiv_t - prev_fdiv
        else:
            fdiv_t, rdiv = 0.0, 0.0
        total = racc + self.lambda1 * rsat + self.lambda2 * rdiv
        return {
            "racc": racc,
            "rsat": rsat,
            "rdiv": rdiv,
            "total": total,
            "fdiv": fdiv_t,
        }
