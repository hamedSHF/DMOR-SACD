"""CLI entry point: train & evaluate DMoR-SACD or one of its baselines.

Usage examples:
    # full DMoR-SACD on MovieLens 100K
    python -m dmorsacd.train --model dmorsacd --dataset ml-100k

    # ablation variant on LastFM with fewer episodes
    python -m dmorsacd.train --model model3-sat --dataset lastfm --episodes 5000

    # quick sanity check
    python -m dmorsacd.train --model dmorsacd --dataset ml-100k --smoke

The trained weights, training log and test metrics are saved under
`runs/<model>_<dataset>/`.
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

from .agents import build_agent
from .config import ALL_MODELS, Config, preset_config
from .data import prepare_data
from .env import RecommendationEnv
from .evaluate import evaluate_agent
from .reward import RewardComponent


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _latest_checkpoint(ckpt_dir: Path):
    """Return the highest-episode checkpoint file, or None."""
    if not ckpt_dir.is_dir():
        return None
    files = list(ckpt_dir.glob("ep*.pt"))
    if not files:
        return None

    def _key(f: Path) -> int:
        try:
            return int(f.stem.replace("ep", ""))
        except ValueError:
            return 0

    return max(files, key=_key)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DMoR-SACD training")
    p.add_argument("--model", type=str, default="dmorsacd", choices=ALL_MODELS,
                   help="model to train (paper name or ablation variant)")
    p.add_argument("--dataset", type=str, default="ml-100k",
                   choices=["ml-100k", "ml-1m", "lastfm", "retailrocket"])
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--max-items", type=int, default=None,
                   help="retailrocket: keep the top-K most frequent items "
                        "(default: config max_items=10000; 0 = keep all)")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--batch-users", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-users", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None,
                   help="print a progress line every N episodes")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run to verify the pipeline")
    p.add_argument("--checkpoint-every", type=int, default=1000,
                   help="save a warm-start checkpoint every N episodes")
    p.add_argument("--resume", action="store_true",
                   help="resume from the latest checkpoint in the run dir "
                        "(warm start for interrupted / long runs)")
    p.add_argument("--save-buffer", action="store_true",
                   help="also save the replay buffer inside checkpoints "
                        "(larger files, but the replay experience is kept)")
    p.add_argument("--save-dir", type=str, default="runs")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in
                 ("model", "dataset", "data_dir", "smoke", "save_dir",
                  "checkpoint_every", "resume", "save_buffer")}
    cfg = preset_config(args.model, dataset=args.dataset,
                        data_dir=args.data_dir, **overrides)
    cfg.smoke = cfg.smoke or args.smoke
    cfg.resolve()
    cfg.save_dir = args.save_dir   # keep config.json honest about the run dir

    set_seed(cfg.seed)
    print(f"[DMoR-SACD] model={args.model} dataset={cfg.dataset} "
          f"device={cfg.device} episodes={cfg.episodes}")

    data = prepare_data(cfg.dataset, cfg.data_dir, seed=cfg.seed,
                        min_interactions=cfg.min_interactions,
                        train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
                        max_items=cfg.max_items)
    print(f"[data] users={data.num_users} items={data.num_items} "
          f"ratings in [{data.rmin}, {data.rmax}] | "
          f"train={len(data.train_users)} val={len(data.val_users)} "
          f"test={len(data.test_users)}")

    reward = RewardComponent(cfg, data)
    env = RecommendationEnv(cfg, data, reward)
    agent = build_agent(cfg, data, env)

    val_users = data.val_users[:cfg.eval_users] if cfg.seed_eval_users \
        else data.val_users
    test_users = data.test_users[:cfg.eval_users] if cfg.seed_eval_users \
        else data.test_users
    if len(val_users) == 0:
        val_users = data.train_users[:cfg.eval_users]

    run_dir = Path(args.save_dir) / f"{args.model}_{cfg.dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- warm start (resume from the latest checkpoint) -------------------
    start_ep = 1
    history = []
    if args.resume:
        ckpt_path = _latest_checkpoint(run_dir / "checkpoints")
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=cfg.device)
            agent.load_state_dict(ckpt)
            start_ep = int(ckpt.get("episode", 0)) + 1
            hist_file = run_dir / "history.json"
            if hist_file.exists():
                history = json.loads(hist_file.read_text())
            print(f"[resume] {ckpt_path.name} -> continuing at episode "
                  f"{start_ep}/{cfg.episodes}")
        else:
            print("[resume] no checkpoint found; starting a fresh run")
        if start_ep > cfg.episodes:
            print(f"[resume] target episodes ({cfg.episodes}) already "
                  f"reached; running the final evaluation only")

    t0 = time.time()   # session start (for the s/ep rate shown below)
    for ep in range(start_ep, cfg.episodes + 1):
        users = np.random.choice(data.train_users, size=cfg.batch_users,
                                 replace=True)
        stats = agent.collect(users)
        for _ in range(cfg.updates_per_episode):
            up = agent.update()
            if up:
                stats.update(up)
        if ep % cfg.log_every == 0:
            print(f"[ep {ep}/{cfg.episodes}] {stats} "
                  f"({(time.time() - t0) / (ep - start_ep + 1):.3f}s/ep)")
        if ep % cfg.eval_every == 0:
            m = evaluate_agent(agent, data, val_users, cfg)
            history.append({"episode": ep, **m})
            print(f"[ep {ep}] val metrics: "
                  + " ".join(f"{kk}={vv:.4f}" for kk, vv in m.items()))
        if ep % args.checkpoint_every == 0:
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            state = agent.state_dict(include_buffer=args.save_buffer)
            state["episode"] = ep
            ckpt_path = ckpt_dir / f"ep{ep}.pt"
            torch.save(state, ckpt_path)
            (ckpt_dir / "state.json").write_text(json.dumps({
                "model": args.model, "dataset": cfg.dataset,
                "episode": ep, "episodes": cfg.episodes,
                "checkpoint": ckpt_path.name,
                "time_s": round(time.time() - t0, 1),
            }, indent=2))
            print(f"[checkpoint] {ckpt_path}")

    # final evaluation on the test users
    test_metrics = evaluate_agent(agent, data, test_users, cfg)
    print("\n[test] " + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    torch.save(agent.net.state_dict(), run_dir / "model.pt")
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "config.json", "w") as f:
        json.dump({k: str(v) for k, v in cfg.__dict__.items()}, f, indent=2)
    print(f"[saved] {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
