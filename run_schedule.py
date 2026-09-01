"""Scheduled runner for the RetailRocket 5000-episode experiments.

Executes the execution plan agreed in the session:

  * dataset        : retailrocket (top-10K candidate items, graded ratings)
  * models         : the "main models" (Rainbow excluded) -- default
                     ``dmorsacd,dqn-r,ddqn``
  * episodes       : 5000 (configurable)
  * batch settings : 64/64 (batch_users/batch_size) -- the CPU-validated
                     settings from the calibration
  * checkpoints    : a warm-start checkpoint is saved every 1000 episodes
                     (``runs/<model>_retailrocket/checkpoints/epN.pt``)
  * warm start     : models whose final checkpoint already exists are
                     skipped; interrupted models auto-resume from their
                     latest checkpoint (pass ``--resume`` internally)
  * parallelism    : up to ``--workers`` training processes run at the same
                     time (default 4), each pinned to
                     ``--threads`` (default ``cores // workers``) CPU threads.

Re-running the same command any time continues from where the previous
session left off -- nothing is ever lost.

Usage:
    python run_schedule.py                          # run the default plan
    python run_schedule.py --dry-run                # print the plan only
    python run_schedule.py --models dmorsacd,ddqn   # subset of models
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dmorsacd.config import ALL_MODELS  # noqa: E402

# Measured per-episode times (s) on this machine at batch 64/64 with
# ~3-4 CPU threads (12-core box). Used only to schedule/balance workers and
# to estimate the wall-clock budget; a 15% safety margin is added.
E_STEP = {
    "dmorsacd": 0.371, "model1": 0.203, "model2": 0.235, "model3": 0.383,
    "model3-sat": 0.365, "model3-div": 0.467, "dqn-r": 0.222, "ddqn": 0.117,
}
SAFETY = 1.15

METRICS = ["R@K", "H@K", "N@K", "D@K"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scheduled RetailRocket experiments")
    p.add_argument("--models", type=str, default="dmorsacd,dqn-r,ddqn",
                   help="comma-separated subset of " + ",".join(ALL_MODELS))
    p.add_argument("--dataset", type=str, default="retailrocket",
                   choices=["ml-100k", "ml-1m", "lastfm", "retailrocket"])
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--batch-users", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-users", type=int, default=100)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=1000,
                   help="save a warm-start checkpoint every N episodes")
    p.add_argument("--workers", type=int, default=4,
                   help="max parallel training processes")
    p.add_argument("--threads", type=int, default=None,
                   help="CPU threads per worker (default: cores // workers)")
    p.add_argument("--save-buffer", action="store_true",
                   help="also save the replay buffer inside checkpoints")
    p.add_argument("--force", action="store_true",
                   help="retrain models that already finished")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and estimated wall clock, don't run")
    p.add_argument("--save-dir", type=str, default="runs")
    return p.parse_args(argv)


def est_minutes(model: str, episodes: int) -> float:
    return E_STEP.get(model, 0.5) * SAFETY * episodes / 60.0


def model_done(model: str, dataset: str, episodes: int,
               save_dir: str) -> bool:
    run_dir = Path(save_dir) / f"{model}_{dataset}"
    final = run_dir / "checkpoints" / f"ep{episodes}.pt"
    return final.exists() and (run_dir / "test_metrics.json").exists()


def latest_episode(model: str, dataset: str, save_dir: str) -> int:
    ckpt_dir = Path(save_dir) / f"{model}_{dataset}" / "checkpoints"
    if not ckpt_dir.is_dir():
        return 0
    best = 0
    for f in ckpt_dir.glob("ep*.pt"):
        try:
            best = max(best, int(f.stem.replace("ep", "")))
        except ValueError:
            pass
    return best


def train_cmd(args, model: str) -> list:
    return [
        sys.executable, "-m", "dmorsacd.train",
        "--model", model,
        "--dataset", args.dataset,
        "--episodes", str(args.episodes),
        "--batch-users", str(args.batch_users),
        "--batch-size", str(args.batch_size),
        "--eval-every", str(args.eval_every),
        "--eval-users", str(args.eval_users),
        "--log-every", str(args.log_every),
        "--checkpoint-every", str(args.checkpoint_every),
        "--resume",
    ] + (["--save-buffer"] if args.save_buffer else [])


def progress_of(args, model: str) -> str:
    """Live progress: latest "[ep N/...]" line in the model's log file;
    falls back to the newest checkpoint episode if no log line yet."""
    ep = 0
    log = Path(args.save_dir) / "logs" / f"{model}.log"
    if log.exists():
        try:
            for ln in reversed(log.read_text(errors="ignore").splitlines()):
                m = re.search(r"\[ep (\d+)/\d+\]", ln)
                if m:
                    ep = int(m.group(1))
                    break
        except OSError:
            pass
    if ep == 0:
        ep = latest_episode(model, args.dataset, args.save_dir)
    pct = 100.0 * ep / args.episodes
    return f"{model:<10} ep {ep}/{args.episodes} ({pct:4.1f}%)"


def main(argv=None) -> None:
    args = parse_args(argv)
    models = [m for m in (x.strip() for x in args.models.split(",")) if m]
    for m in models:
        if m not in ALL_MODELS:
            raise SystemExit(f"Unknown model '{m}'. Choose from {ALL_MODELS}.")

    threads = args.threads or max(2, os.cpu_count() // args.workers)
    cores = os.cpu_count()
    log_dir = Path(args.save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "schedule.log"

    # ---------- plan ----------
    print("=" * 78)
    print(f"[schedule] dataset={args.dataset} episodes={args.episodes} "
          f"batch={args.batch_users}/{args.batch_size} "
          f"checkpoint-every={args.checkpoint_every}")
    print(f"[schedule] workers={min(args.workers, len(models))} "
          f"threads/worker={threads} (cores={cores})")
    print("-" * 78)
    print(f"{'model':<12}{'est. min':>10}{'status':>20}")
    pending, total_est = [], 0.0
    for m in models:
        done = model_done(m, args.dataset, args.episodes, args.save_dir)
        if args.force:
            done = False
        if done:
            status = "done (skip)"
        else:
            ep = latest_episode(m, args.dataset, args.save_dir)
            status = f"resume ep {ep}" if ep > 0 else "fresh"
            pending.append(m)
        em = est_minutes(m, args.episodes)
        total_est += em
        print(f"{m:<12}{em:>8.1f} min{status:>20}")
    print("-" * 78)
    n_workers = max(1, min(args.workers, len(pending)))
    if pending:
        ideal = total_est / n_workers
        print(f"est. sequential {total_est:.0f} min -> with {n_workers} "
              f"workers ~{ideal:.0f} min wall (load balanced)")
    print("=" * 78)

    if args.dry_run:
        return
    if not pending:
        print("[schedule] all models already finished -- nothing to do; "
              "showing last results below.")
        _print_results(args)
        return

    # ---------- run (greedy load-balancing over the estimated times) -------
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"

    queue = sorted(pending, key=lambda m: -est_minutes(m, args.episodes))
    running: dict = {}            # model -> Popen
    finished: dict = {}           # model -> exit code
    t_start = time.time()
    with open(log_file, "a", buffering=1) as slog:
        slog.write(f"\n--- run started {time.ctime()} | models={models} "
                   f"threads={threads} ---\n")
        while queue or running:
            while queue and len(running) < args.workers:
                m = queue.pop(0)
                mlog = open(log_dir / f"{m}.log", "a", buffering=1)
                mlog.write(f"\n--- {m} started {time.ctime()} "
                           f"(from ep {latest_episode(m, args.dataset, args.save_dir)}) ---\n")
                mlog.flush()
                proc = subprocess.Popen(train_cmd(args, m), cwd=str(HERE),
                                        env=env, stdout=mlog, stderr=subprocess.STDOUT)
                running[m] = (proc, mlog)
                slog.write(f"started {m} (pid {proc.pid})\n")
                print(f"[started] {m} (pid {proc.pid})", flush=True)

            time.sleep(20)
            done_models = [m for m, (p, _) in running.items() if p.poll() is not None]
            if done_models:
                for m in done_models:
                    proc, mlog = running.pop(m)
                    mlog.close()
                    finished[m] = proc.returncode
                    slog.write(f"finished {m} rc={proc.returncode}\n")
                    if proc.returncode != 0:
                        print(f"[WARN] {m} exited with rc={proc.returncode} "
                              f"(see runs/logs/{m}.log); it will resume "
                              f"from its latest checkpoint on the next run",
                              flush=True)
                    else:
                        print(f"[finished] {m} (rc=0, "
                              f"{(time.time() - t_start) / 60:.1f} min)",
                              flush=True)
            else:
                line = "  ".join(progress_of(args, m) for m in running)
                print(f"  [{elapsed_min(t_start):5.1f} min] {line}", flush=True)
                slog.write(f"t+{elapsed_min(t_start):.1f} min | {line}\n")

    _print_results(args)
    print(f"[log]   {log_file}")


def _print_results(args) -> None:
    """Aggregate test_metrics.json of every requested model + print table."""
    results = {}
    for m in (x.strip() for x in args.models.split(",") if x.strip()):
        met_file = Path(args.save_dir) / f"{m}_{args.dataset}" / "test_metrics.json"
        if met_file.exists():
            results[m] = json.loads(met_file.read_text())
    out = Path(args.save_dir) / \
        f"all_models_results_{args.episodes}ep_{args.dataset}.json"
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    header = f"{'Method':<12}" + "".join(f"{mk:>10}" for mk in METRICS)
    print(header)
    print("-" * 78)
    for m in (x.strip() for x in args.models.split(",") if x.strip()):
        r = results.get(m, {})
        row = f"{m:<12}" + "".join(
            f"{r.get(mk, float('nan')):>10.4f}" for mk in METRICS)
        print(row)
    print("=" * 78)
    print(f"[saved] {out}")


def elapsed_min(t0: float) -> float:
    return (time.time() - t0) / 60.0


if __name__ == "__main__":
    sys.exit(main())
