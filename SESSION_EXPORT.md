# Session Export — DMoR-SACD investigation

**Date:** 2026-08-07
**Project root:** `C:\Users\nt924\Desktop\RS limitations`
**Package under study:** `RL in RS/MultiObjective RL/DMoR-SACD/` (PyTorch implementation of Luo, Li & Jiao 2025, "A dynamic multiobjective recommendation method based on soft actor-critic with discrete actions", JKSU-CIS, doi:10.1007/s44443-025-00016-3)

> This file is a handoff so a fresh session can resume without re-reading the
> original conversation. It records: what was asked, what was verified, the
> decisions the user made, and the exact state of an interrupted run.

---

## 1. Environment (verified)

- Windows 10/11, bash shell (POSIX syntax, forward-slash paths).
- Python 3.10.6, torch 2.2.0 **+cpu** (no CUDA), numpy/pandas/scipy installed.
- Everything below ran on **CPU only**. `torch.set_num_threads(8)` was used in benchmarks.

## 2. The user's three questions + answers

### Q1: Can you run this code on a little subset of datasets? — **Yes, verified ✅**

- CLI knobs: `--smoke` (60 episodes, batch_users 128, batch_size 64, replay 5k), `--episodes N`, `--batch-users`, `--eval-users`, `--eval-every`.
- Verified: `python -m dmorsacd.train --model dmorsacd --dataset ml-100k --smoke` → **~25 s** (0.42 s/ep at smoke settings; 0.27–0.30 s/ep at batch 64/64). Full pipeline (train → val → test → save under `runs/dmorsacd_ml-100k/`) works.
- ⚠️ Metrics are meaningless at tiny episode counts (paper trained ~150k episodes).

### Q2: Can you run the ablation models too? — **Yes, all 9 verified ✅**

- `run_all.py` trains all 9 presets (defined in `dmorsacd/config.py` → `MODEL_PRESETS`):
  `dmorsacd, model1, model2, model3, model3-sat, model3-div, dqn-r, ddqn, rainbow`.
- Verified smoke table (60 eps, ml-100k) saved at `runs/all_models_results.json`:

```
Method             R@K       H@K       N@K       D@K
dmorsacd      -18.7000    0.0490    0.0458    0.7296
model1        -18.7500    0.0550    0.0512    0.8206
model2        -18.7500    0.0550    0.0512    0.8206
model3        -18.7000    0.0490    0.0458    0.7296
model3-sat    -18.7000    0.0490    0.0458    0.7296
model3-div    -18.7000    0.0490    0.0458    0.7296
dqn-r         -17.0100    0.1110    0.1206    0.7173
ddqn          -16.6500    0.1180    0.1654    0.7078
rainbow       -18.2700    0.0680    0.0905    0.6882
```

- Note: at 60 eps, DMoR-SACD / Model3 / 3-sat / 3-div rows are identical (same GRU encoder + SACD; λ=1/|A|≈6e-4 makes sat/div reward terms tiny; too few episodes for differences to emerge).

### Q3: Can these models work on RetailRocket? — **Yes, but needs a data loader; algorithm code needs ZERO changes**

**RetailRocket stats** (from `EDA-RS datasets/RetailRocket dataset/`):
- `events.csv`: 2,756,101 events — 2,664,312 view / 69,332 addtocart / 22,457 transaction; 1,407,580 visitors; 235,061 items. Implicit feedback, **no ratings**.
- `item_properties_part1/2.csv`: `timestamp,itemid,property,value`; `categoryid` covers ~89k items in first 4M rows; 1,117 unique leaf categories.
- `category_tree.csv`: `categoryid,parentid`.

**User decisions (made via ask_user in the original session):**
1. Rating mapping: **Graded — view=1, addtocart=2, transaction=3** (max per user-item). This keeps `positive_threshold=3.0` unchanged: only purchases count as "positive", ratings normalize to view→−1, addtocart→0, transaction→+1.
2. Item space: **top ~10K most frequent items** (tractable on CPU; standard recsys candidate-set practice).
3. Scope: **analysis only — no project code changes** (do not implement the loader unless the user later asks).

**Simulated data prep** (top-10K items → then ≥20-interaction user filter, seed 42):
- **585 users / 7,442 items / 42,624 user-item pairs** (smaller than ml-100k!)
- rating distribution: {1.0: 37,289, 2.0: 1,849, 3.0: 3,486} → **8.2% positive**
- 80/10/10 split → 468 / 58 / 59 users
- 97.7% of kept items have a categoryid (587 distinct leaf categories) → good item-vector coverage

**Network cost benchmark (actual dmorsacd networks, CPU):**

| Actions | SACD params/RAM | SACD est. s/ep | DQN | Rainbow params/RAM | Rainbow est. s/ep |
|---|---|---|---|---|---|
| 1,682 (ml-100k) | 1.1M / 4 MB | ~0.4 | 0.2M | 8.9M / 35 MB | ~3 |
| 10,000 | 5.3M / 21 MB | ~1–2 | 1.1M | 52.5M / 210 MB | ~23.5 |
| 235,061 (full) | 118.9M / 476 MB | ~15 | — | ~1.2B (impractical) | — |

**The three hurdles (why it's not plug-and-play):**
1. **Format:** `data.py` supports only ml-100k / ml-1m / lastfm and requires explicit ratings + item vectors. RetailRocket needs a loader (~60 lines in `data.py` + registering `"retailrocket"` in `config.py`). ⚠️ Do NOT use the generic column-vector fallback — 235k items × 1.4M users = **1.3 TB**. Must build category-based vectors from `item_properties` (97.7% coverage).
2. **Action space:** 235K-wide output head → 476 MB SACD, ~15 s/ep; Rainbow's advantage head (|A|×51) is impractical. Restrict to a candidate item set (user chose top-10K).
3. **Methodological:** λ₁=λ₂=1/|A| is calibrated for |A|≈1,682 (λ≈6e-4). At 7.4K items λ≈1.3e-4; at 235K λ≈4e-6 → satisfaction/diversity rewards vanish → DMoR-SACD degenerates to pure accuracy. Rescaling λ (e.g. 1/√|A|) would be needed at scale.

**Verdict:** workable after a loader + config registration; models/agents/env/reward/evaluate need no changes. Expect low H@K/N@K ceilings (sparse positives).

---

## 3. Interrupted run: all-9-models, 2000 episodes on ml-100k ⚠️

User asked (after the Q1–Q3 report): *"Run all 9 models on ml-100k with more episodes (e.g. 2000) so the ablation differences actually separate, and show the comparison table."*

**What exists:**
- Temp runner at project root: **`temp_run_all_2000.py`** (imports the real `dmorsacd` package; mirrors `run_all.py`).
  - Usage: `python temp_run_all_2000.py [model...]` (default: all 9); env var `DMOR_EPISODES=N` overrides the default 2000.
  - Settings: episodes=2000, **batch_users=64, batch_size=64**, eval_users=50, eval_every=500, seed=42, data_dir=`RL in RS/MultiObjective RL/DMoR-SACD/data`.
  - Prints a comparison table and saves `RL in RS/MultiObjective RL/DMoR-SACD/runs/all_models_results_2000ep.json` at the end.
- Calibration (30 eps): dmorsacd 0.27 s/ep → **~9–10 min per SACD model**; dqn-r ~0.2 s/ep → ~7 min; rainbow 1.43 s/ep → **~48 min**. Total ≈ **2 h sequential on CPU**.

**Run state:**
- The real 2000-ep run was **aborted by the user mid-way**: dmorsacd had reached ep 600/2000 (0.30 s/ep), then the user interrupted.
- **No results from that run were saved** (the runner only writes JSON when ALL requested models finish).
- ⚠️ `runs/all_models_results_2000ep.json` currently contains only the **30-episode calibration** numbers (dmorsacd R@K −18.35 / dqn-r −17.97 / rainbow −18.32) — do not mistake it for the real run; it gets overwritten by the next full run.
- No per-model weights were saved by the runner (it only evaluates + writes JSON; use `dmorsacd.train` for weight checkpoints).

**To resume (next session):**
```bash
cd "C:/Users/nt924/Desktop/RS limitations"
python temp_run_all_2000.py dmorsacd model1 model2 model3 model3-sat model3-div dqn-r ddqn rainbow   # ~2 h
# or a subset:  python temp_run_all_2000.py dmorsacd model1 model2
```
Each model group is idempotent/independent; results JSON is written once all requested models finish. For progress visibility, each model prints `[name] ep N/2000 (s/ep)` every 200 episodes.

## 4. Artifacts currently on disk

| Path | What it is |
|---|---|
| `RL in RS/MultiObjective RL/DMoR-SACD/runs/dmorsacd_ml-100k/` | smoke-run artifacts (model.pt, test_metrics.json, history.json, config.json) |
| `RL in RS/MultiObjective RL/DMoR-SACD/runs/all_models_results.json` | smoke table, 60 eps, all 9 models |
| `RL in RS/MultiObjective RL/DMoR-SACD/runs/all_models_results_2000ep.json` | ⚠️ calibration only (30 eps, 3 models) — will be overwritten |
| `temp_run_all_2000.py` | temp runner (keep while resuming; safe to delete after the real run) |

**Project code was NOT modified** in this session (per the user's "analysis only" choice). All temp analysis scripts from Q3 (stats/prep/benchmark) were deleted; only `temp_run_all_2000.py` remains.

## 5. Useful commands (from `RL in RS/MultiObjective RL/DMoR-SACD`)

```bash
python -m dmorsacd.train --model dmorsacd --dataset ml-100k --smoke          # sanity (~25 s)
python -m dmorsacd.train --model model3-sat --dataset lastfm --episodes 5000 # single model
python run_all.py --dataset ml-100k --episodes 2000                          # all 9 (default batches, slower)
```

## 6. Candidate next steps

1. Resume the 2000-episode all-9 run (see §3) and show the comparison table.
2. ~~Implement the RetailRocket loader~~ → **DONE, see §7**.
3. Investigate rescaling λ=1/|A| (e.g. 1/√|A|) so satisfaction/diversity rewards stay meaningful at 7.4K–235K action spaces.
4. Estimate GPU budgets for the full 235K-item RetailRocket space.
5. Train RetailRocket models for real (e.g. 2K+ episodes) and compare against ml-100k.

---

## 7. RetailRocket loader — IMPLEMENTED (2026-08-07, after this export)

Per the agreed design, the loader was implemented in the project (no longer
analysis-only):

**Code changes:**
- `dmorsacd/config.py`: added `RETAILROCKET = "retailrocket"` and
  `max_items: int = 10_000` (candidate-item-set cap; 0 = keep all).
- `dmorsacd/data.py`: added `_load_retailrocket` (graded view=1/addtocart=2/
  transaction=3, max per user-item; top-K item filter by pair frequency),
  `_category_vectors` (one-hot leaf-category item vectors from
  `item_categories.csv`, with raw `item_properties_part*.csv` fallback),
  `max_items` param on `prepare_data` + dispatch + vector branch.
- `dmorsacd/train.py` + `run_all.py`: registered `"retailrocket"` in
  argparse choices; pass `max_items=cfg.max_items` to `prepare_data`.
- `README.md`: short RetailRocket section.

**Data staged** at `data/retailrocket/`: `events.csv` (94 MB, copied) +
`item_categories.csv` (5.1 MB, preprocessed from the 900 MB raw properties
once via a now-deleted temp script; 417,053 items, 1,218 categories) +
`category_tree.csv`. The raw 900 MB `item_properties_part*.csv` were NOT
copied — the loader falls back to them only if the compact file is absent.

**Verified smoke results on RetailRocket** (top-10K items → 585 users /
7,442 items / 42,624 pairs; all at 60 episodes unless noted):

| Model | R@K | H@K | N@K | D@K |
|---|---|---|---|---|
| dmorsacd | −19.92 | 0.014 | 0.010 | 0.9998 |
| model3-sat | −19.92 | 0.014 | 0.010 | 0.9998 |
| dqn-r | −19.88 | 0.019 | 0.017 | 0.984 |
| ddqn | −20.00 | 0.024 | 0.025 | 0.990 |
| rainbow (30 eps) | −19.96 | 0.011 | 0.014 | 0.995 |

- Runtime: dmorsacd ≈ 0.72 s/ep on CPU (7,442 actions, smoke batches).
- H@K ~0.01–0.02 at smoke scale is expected: only 8.2% of pairs are
  positive and 60–30 episodes is far below the paper's ~150k.
- Regression: ml-100k smoke results are byte-identical to pre-change
  (R@K=−18.70, H@K=0.049, D@K=0.7296) → existing datasets unaffected.
- Rainbow is slow at 7,442 actions (its smoke run needs ~11 min, longer
  than the 10-min command cap — use fewer episodes for quick checks).

**Commands:**
```bash
python -m dmorsacd.train --model dmorsacd --dataset retailrocket --smoke
python run_all.py --dataset retailrocket --smoke --models dqn-r,ddqn
```

Run artifacts under `runs/`: `dmorsacd_retailrocket/`, `dqn-r_retailrocket/`,
`ddqn_retailrocket/`, `model3-sat_retailrocket/`, `rainbow_retailrocket/`.

---

## 8. RetailRocket 5000-episode schedule — IMPLEMENTED & LAUNCHED (2026-08-07)

### What the user asked (this session)

1. Schedule the **main models on RetailRocket**, **Rainbow excluded**.
2. Everything must finish **within 2 hours**.
3. `episodes = 5000`, save parameters **every 1000 episodes** so later
   sessions can **warm start** (resume).
4. Keep all history/changes of the chat in this file.

### Decisions (via ask_user)

- **Models:** `dmorsacd, dqn-r, ddqn` (the 3 core models).
- **Execution:** up to **4 parallel worker processes**, batch **64/64**
  (batch_users=64, batch_size=64) — the CPU-validated settings.
- Rainbow was excluded as requested (impractical on CPU: ~6.6 s/ep at
  64/64 → ~550 min for 5000 eps).

### Measured timing (this session, 12-core box, batch 64/64)

| Model | isolated ~3-4 thr (s/ep) | in-run 3-way parallel (s/ep) | 5000-ep est. |
|---|---|---|---|
| dmorsacd | 0.371 | **0.76** | ~64 min |
| dqn-r    | 0.222 | **0.41** | ~34 min |
| ddqn     | 0.117 | **0.22** | ~18 min |

All 3 run concurrently → **wall ≈ 65–70 min**, well inside the 2 h cap.
(Contention makes in-run rates ~2× the isolated calibration; the runner's
estimates use isolated numbers +15% and are therefore optimistic.)

### Code changes (this session)

1. **`dmorsacd/agents.py`** — added `state_dict(include_buffer=False)` /
   `load_state_dict(d)` to `SACDAgent`, `DQNAgent`, `RainbowAgent`:
   full trainer state = online+target networks, optimizers, `log_alpha`
   (SACD), `step_count`, `beta` (Rainbow), and the replay buffer
   (optional). A `[warn]` fires if the checkpoint's agent type mismatches.
2. **`dmorsacd/train.py`** — new CLI flags:
   - `--batch-size`, `--log-every` (progress cadence)
   - `--checkpoint-every N` (default 1000): saves a full warm-start
     checkpoint `runs/<model>_<dataset>/checkpoints/ep<N>.pt` + a tiny
     `state.json` every N episodes
   - `--resume`: auto-picks the **latest** checkpoint and continues from
     `ep+1`, re-loading `history.json` so val history survives resumption
   - `--save-buffer`: also store the replay buffer in checkpoints (bigger)
   - **bugfix:** `--save-dir` was previously ignored (cfg field excluded
     from overrides); it now actually relocates the run dir
3. **`run_schedule.py`** (NEW) — the execution plan itself:
   - defaults: retailrocket, 5000 eps, 64/64, checkpoint-every 1000,
     eval-every 1000, workers=4 (threads = cores//workers)
   - launches one `python -m dmorsacd.train ... --resume` subprocess per
     model (logs → `runs/logs/<model>.log`, schedule log →
     `runs/logs/schedule.log`), max `--workers` at a time
   - **idempotent**: skips models with a final `ep5000.pt` +
     `test_metrics.json`; resumes interrupted ones; `--force` retrains
   - prints a plan table (`--dry-run`), live progress, and an aggregated
     R@K/H@K/N@K/D@K table into
     `runs/all_models_results_5000ep_retailrocket.json`

### Validation (all green)

- Checkpoint→resume roundtrips on RetailRocket for **ddqn** (DQN path) and
  **dmorsacd** (SACD path): resume correctly continues at `ep+1`.
- History preserved across resume (`history.json` had entries from both
  runs).
- `python -m py_compile` clean on all changed files.
- `python run_schedule.py --dry-run` prints the expected plan.

### Run state at handoff

- Launched detached (`nohup python run_schedule.py > runs/schedule_5000.log`)
  at ~20:07. All 3 models running as separate processes; per-model logs in
  `runs/logs/`.
- Progress observed up to session end:
  - ddqn ≈ ep 1600/5000 (0.22 s/ep), first checkpoint `ep1000.pt` saved ✓
  - dqn-r ≈ ep 800/5000 (0.41 s/ep)
  - dmorsacd ≈ ep 400/5000 (0.76 s/ep)
- Checkpoints (nets + optimizers + alpha, ~90 MB each) appear every 1000
  eps under `runs/<model>_retailrocket/checkpoints/`.

### Commands

```bash
# check live progress (runner + per-model tails)
tail -f "RL in RS/MultiObjective RL/DMoR-SACD/runs/schedule_5000.log"
tail -f "RL in RS/MultiObjective RL/DMoR-SACD/runs/logs/dmorsacd.log"

# resume / continue after any interruption (safe to re-run at any time)
cd "RL in RS/MultiObjective RL/DMoR-SACD" && python run_schedule.py

# preview the plan without running
python run_schedule.py --dry-run

# results table after the run
python run_schedule.py --dry-run   # prints the saved table too
cat runs/all_models_results_5000ep_retailrocket.json

# single model manually with warm start
python -m dmorsacd.train --model dmorsacd --dataset retailrocket \
  --episodes 5000 --batch-users 64 --batch-size 64 --checkpoint-every 1000 --resume
```

### Next steps

1. When the run finishes (~70 min wall), collect the comparison table.
2. Extend the schedule to the remaining baselines (model1/2/3, 3-sat,
   3-div) in a later session — checkpoints make each model independent.
3. Consider `λ = 1/√|A|` rescaling for satisfaction/diversity rewards at
   the 7.4K-item action space (see §2 Q3 hurdle 3).
