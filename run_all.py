"""Run every DMoR-SACD variant and ablation baseline on one dataset and
print a comparison table (mirrors Tables 4 & 5 of the paper).

Usage:
    python run_all.py --dataset ml-100k --episodes 2000
    python run_all.py --dataset ml-100k --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dmorsacd.agents import build_agent                      # noqa: E402
from dmorsacd.config import ALL_MODELS, preset_config        # noqa: E402
from dmorsacd.data import prepare_data                       # noqa: E402
from dmorsacd.env import RecommendationEnv                   # noqa: E402
from dmorsacd.evaluate import evaluate_agent                 # noqa: E402
from dmorsacd.reward import RewardComponent                  # noqa: E402
from dmorsacd.train import set_seed                          # noqa: E402

METRICS = ["R@K", "H@K", "N@K", "D@K"]


def train_one(model: str, cfg, data, env) -> dict:
    set_seed(cfg.seed)
    agent = build_agent(cfg, data, env)
    val_users = data.val_users[:cfg.eval_users] if cfg.seed_eval_users \
        else data.val_users
    test_users = data.test_users[:cfg.eval_users] if cfg.seed_eval_users \
        else data.test_users
    t0 = time.time()
    for ep in range(1, cfg.episodes + 1):
        users = np.random.choice(data.train_users, size=cfg.batch_users,
                                 replace=True)
        agent.collect(users)
        for _ in range(cfg.updates_per_episode):
            agent.update()
        if ep % cfg.eval_every == 0:
            evaluate_agent(agent, data, val_users, cfg)   # checkpoint eval
        if ep % max(1, cfg.episodes // 10) == 0:
            print(f"  [{model}] ep {ep}/{cfg.episodes} "
                  f"({(time.time() - t0) / ep:.2f}s/ep)", flush=True)
    m = evaluate_agent(agent, data, test_users, cfg)
    m["time_s"] = time.time() - t0
    return m


def main() -> None:
    p = argparse.ArgumentParser(description="Train all DMoR-SACD baselines")
    p.add_argument("--dataset", type=str, default="ml-100k",
                   choices=["ml-100k", "ml-1m", "lastfm", "retailrocket"])
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-items", type=int, default=None,
                   help="retailrocket: keep the top-K most frequent items "
                        "(default: config max_items=10000; 0 = keep all)")
    p.add_argument("--models", type=str, default=",".join(ALL_MODELS),
                   help="comma-separated subset of " + ",".join(ALL_MODELS))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--data-dir", type=str, default="data")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    base_kw = dict(dataset=args.dataset, data_dir=args.data_dir,
                   episodes=args.episodes)
    if args.max_items is not None:
        base_kw["max_items"] = args.max_items
    cfg = preset_config(models[0], **base_kw)
    cfg.smoke = cfg.smoke or args.smoke
    cfg.resolve()

    print(f"[run_all] dataset={cfg.dataset} episodes={cfg.episodes} "
          f"device={cfg.device} models={models}")

    data = prepare_data(cfg.dataset, cfg.data_dir, seed=cfg.seed,
                        min_interactions=cfg.min_interactions,
                        train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
                        max_items=cfg.max_items)
    print(f"[data] users={data.num_users} items={data.num_items}")

    results = {}
    for model in models:
        mkw = dict(dataset=cfg.dataset, data_dir=cfg.data_dir,
                   episodes=args.episodes)
        if args.max_items is not None:
            mkw["max_items"] = args.max_items
        mcfg = preset_config(model, **mkw)
        mcfg.smoke = args.smoke
        mcfg.resolve()
        reward = RewardComponent(mcfg, data)
        env = RecommendationEnv(mcfg, data, reward)
        print(f"\n=== training {model} ===", flush=True)
        results[model] = train_one(model, mcfg, data, env)
        print(f"  -> " + " ".join(f"{k}={v:.4f}" for k, v in
                                  results[model].items() if k != "time_s"),
              flush=True)

    # print comparison table
    print("\n" + "=" * 64)
    header = f"{'Method':<12}" + "".join(f"{m:>10}" for m in METRICS)
    print(header)
    print("-" * 64)
    for model in models:
        m = results[model]
        row = f"{model:<12}" + "".join(f"{m[kk]:>10.4f}" for kk in METRICS)
        print(row)
    print("=" * 64)

    out = Path("runs") / "all_models_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    sys.exit(main())
