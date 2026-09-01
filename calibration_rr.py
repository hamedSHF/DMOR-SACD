"""Temp timing calibration: s/episode for every model on RetailRocket.

Runs `EPISODES` training episodes (no evaluation) per model at batch 64/64
and prints the per-episode time. Delete after use.

Usage:
    python calibration_rr.py [episodes] [model ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from dmorsacd.agents import build_agent
from dmorsacd.config import ALL_MODELS, preset_config
from dmorsacd.data import prepare_data
from dmorsacd.env import RecommendationEnv
from dmorsacd.reward import RewardComponent
from dmorsacd.train import set_seed

DEFAULTS = "--defaults" in sys.argv
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MODELS = [a for a in sys.argv[2:] if a != "--defaults"] or ALL_MODELS

t_load = time.time()
data = prepare_data("retailrocket", "data", seed=42,
                    min_interactions=20, max_items=10_000)
print(f"[data] users={data.num_users} items={data.num_items} "
      f"(load {time.time() - t_load:.1f}s)", flush=True)

times = {}
for model in MODELS:
    cfg = preset_config(model, dataset="retailrocket", data_dir="data",
                        episodes=EPISODES)
    if DEFAULTS:
        cfg.eval_every = 2000              # default eval cadence (no evals)
    else:
        cfg.batch_users = 64
        cfg.batch_size = 64
        cfg.eval_users = 50
        cfg.eval_every = EPISODES + 1      # no evaluation during calibration
    set_seed(cfg.seed)
    agent = build_agent(cfg, data, RecommendationEnv(cfg, data,
                                                     RewardComponent(cfg, data)))
    t0 = time.time()
    for _ in range(EPISODES):
        users = np.random.choice(data.train_users, size=cfg.batch_users,
                                 replace=True)
        agent.collect(users)
        for _ in range(cfg.updates_per_episode):
            agent.update()
    dt = (time.time() - t0) / EPISODES
    times[model] = round(dt, 4)
    print(f"{model:<12} {dt:.3f} s/ep   "
          f"-> 2000ep {dt*2000/60:6.1f} min | 5000ep {dt*5000/60:6.1f} min "
          f"| 20000ep {dt*20000/60:6.1f} min", flush=True)

print(json.dumps(times, indent=2))
