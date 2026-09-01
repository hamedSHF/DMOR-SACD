# DMOR-SACD
Code for dynamic multiobjective recommendation method based onsoft actor-critic with discrete actions
=======
# DMoR-SACD

A PyTorch implementation of

> **A dynamic multiobjective recommendation method based on soft actor-critic
> with discrete actions**
> Jun Luo, Fenglian Li & Jiangli Jiao, *Journal of King Saud University –
> Computer and Information Sciences* (2025) 37:1,
> https://doi.org/10.1007/s44443-025-00016-3

The paper's DMoR-SACD model **and all of its ablation-study baselines**
(DQN-R, DDQN, Rainbow, Model 1/2/3, Model 3-sat, Model 3-div) are
implemented, trainable and evaluated offline on MovieLens 100K / 1M and
LastFM with the paper's four metrics (R@K, H@K, N@K, D@K).

RetailRocket (implicit feedback) is also supported: the loader converts
events to pseudo-ratings (view=1, addtocart=2, transaction=3, max per
user-item) and builds category-based item vectors. Because the full item
space has ~235K items, `cfg.max_items` (default 10,000) keeps the top-K
most frequent items as the candidate set.

```bash
# run DMoR-SACD on RetailRocket (data/retailrocket/ must contain events.csv
# and item_categories.csv; the loader falls back to the raw
# item_properties_part*.csv files if the compact mapping is absent)
python -m dmorsacd.train --model dmorsacd --dataset retailrocket --smoke
```

---

## Method (summary)

The recommendation problem is modelled as an MDP `⟨S, A, T, R, γ⟩`.

**State component (Eq. 2–3).** A state is `s_t = {s+_t, s-_t}`, the last
`N = 10` positively / negatively received items. Two GRUs extract positive
and negative preference features, two MLPs refine them, and a final MLP
fuses them into the state feature `s'_t`.

**Equivalence transformation (Section 3.2.2).** The state is decomposed into
the *item sequence* `{a_1..a_t}` and the *interest code* `v_i` (`+1`/`-1`/`0`
per step), which feed the reward component.

**Reward component (Eq. 4–9).**
`R(s_t, a_t) = r_acc + λ₁·r_sat + λ₂·r_div`, with

* `r_acc` – rating normalised to [-1, 1] (unrated items → -1);
* `r_sat` – satisfaction derived from the interest code, `f_sat(v_i)/K`;
* `r_div` – diversity via a length-10 sliding window and cosine similarity,
  `r_div = f_div(a_t) − f_div(a_{t−1})`, switched on once the window is full;
* `λ₁ = λ₂ = 1/|A|`.

**Policy component (SACD, Section 3.4).** The actor outputs a categorical
distribution over all `|A|` items; two critics output soft Q-values for every
item; two target critics provide `min(Q'_1, Q'_2)`. Training maximises the
entropy-augmented objective with an adaptively tuned temperature `α`
(target entropy `H̄ = −0.8·log(1/|A|)`).

Hyper-parameters (paper Section 4.1): `N = 10`, `K = 20`, embedding/hidden
size 50, `γ = 0.9`, lr `2.5e-4`, batch 256, replay 1e5, `τ = 0.01`, He init.

## Project layout

```
DMoR-SACD/
├── data/
│   └── download_datasets.py        # fetch ML-100K / ML-1M / LastFM
├── dmorsacd/
│   ├── config.py                   # hyper-parameters + model presets
│   ├── data.py                     # loaders, 80/10/10 splits, item vectors
│   ├── reward.py                   # accuracy / satisfaction / diversity
│   ├── env.py                      # offline simulator (states, interest code)
│   ├── models.py                   # GRU state component, SACD/DQN/Rainbow nets
│   ├── buffers.py                  # replay + prioritised replay
│   ├── agents.py                   # SACD, DQN-R/DDQN, Rainbow trainers
│   ├── evaluate.py                 # R@K, H@K, N@K, D@K
│   └── train.py                    # CLI
├── run_all.py                      # train everything, print tables
└── README.md
```

## Model → ablation mapping

| Paper name      | State encoder | Reward             | Agent  |
|-----------------|---------------|--------------------|--------|
| `dmorsacd`      | two GRUs      | acc + sat + div    | SACD   |
| `model1`        | pooled seq    | accuracy           | SACD   |
| `model2`        | pooled seq    | acc + sat + div    | SACD   |
| `model3`        | two GRUs      | accuracy           | SACD   |
| `model3-sat`    | two GRUs      | acc + satisfaction | SACD   |
| `model3-div`    | two GRUs      | acc + diversity    | SACD   |
| `dqn-r`         | two GRUs      | accuracy           | DQN (ε=0.1) |
| `ddqn`          | pooled seq    | accuracy           | Double DQN |
| `rainbow`       | pooled seq    | accuracy           | Rainbow (dueling + C51 + noisy + PER + n-step) |

## Quick start

```bash
pip install -r requirements.txt

# 1) download data
python data/download_datasets.py ml-100k        # ~5 MB

# 2) sanity check (tiny run, ~1 min on CPU)
python -m dmorsacd.train --model dmorsacd --dataset ml-100k --smoke

# 3) real training
python -m dmorsacd.train --model dmorsacd --dataset ml-100k --episodes 20000

# 4) train all variants & baselines, print comparison table
python run_all.py --dataset ml-100k --episodes 2000
```

Results (model weights, test metrics, config) are saved under `runs/`.

## Notes & faithful-reproduction details

* Users with fewer than 20 interactions are dropped; users are split
  80/10/10; each user's full rating record is the offline environment.
* During an episode an item is never recommended twice (paper Section 3.5).
* Training actions are sampled from the actor's categorical distribution;
  evaluation uses the max-probability / max-Q item (Algorithms 1 & 2).
* Item similarity (diversity reward + D@K) uses genre vectors for
  MovieLens (content diversity, as motivated in the paper) and normalised
  play-count columns for LastFM.
* Reward-mode variants are realised through `RewardComponent`; the
  satisfaction term counts positive/negative runs of length ≥ 3 in the
  interest code (Section 3.3.2).
* On CPU, `--episodes 20000` on ML-100K takes a few hours; use fewer
  episodes or `run_all.py --smoke` for quick experiments. The paper trained
  ~150k episodes.
