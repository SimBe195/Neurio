# Neurio

A reinforcement learning agent that learns to play Super Mario Bros from the screen, using PPO. The goal is an agent
that gets good at the game on its own: no human demonstrations and no level-specific scripting.

## Approach

**Environment.** [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) emulates the NES game through
[gymnasium](https://gymnasium.farama.org). 32 copies run in parallel worker processes. Each observation is processed as
follows:

- converted to grayscale
- cropped to remove the score bar and borders
- downscaled by 2, to 94×114 pixels
- stacked with the previous 3 frames, so the agent can see motion

Each chosen action is held for 4 frames. The agent picks from the 12 button combinations of `COMPLEX_MOVEMENT`.

**Reward.** The game's own reward, scaled by 0.1: progress to the right, a small time penalty, points, coins, power-ups,
+50 for the flag and −25 for dying. Episode rewards in the metrics are always this reward, without the curiosity
bonus below, so runs with different settings are comparable.

**Agent.** PPO with generalized advantage estimation (GAE) and gamma 0.95. The IMPALA residual encoder (3 stages
with 32/64/64 channels and a 256-unit layer) reads the frames; it generalizes better to unseen levels than the DQN
encoder ("Nature CNN", `agent/model=nature`). On the GPU the encoder runs in bfloat16. Together with an embedding of
the previous action, it feeds two outputs: action logits (the actor) and a state value (the critic). Each iteration
collects 128 steps per worker and trains 5 epochs on them. Advantages are normalized per minibatch. The entropy bonus
decays from 0.013 to 0.001 over training.

**Curiosity.** `exploration.bonus_scale` (default 0.7, 0 disables it) pays a bonus of `scale / √visits` for reaching a
position (level, 16-pixel x and y tile). Visits are counted across all workers. This lets the agent find its way
around dead ends that the game's reward makes attractive. Example: on 1-3, a jump into a pit earns more progress
reward than dying costs. The bonus is only used during training.

**Self-imitation.** `agent.sil_loss_weight` (default 0.7, 0 disables it): the agent keeps its own 5 best episodes per
level (by game reward, `agent.sil_episodes_per_level`) and, after each update, imitates their actions wherever their
return beats its current value estimate. Curiosity makes the agent find better routes; self-imitation makes it learn
them before it commits to a worse one.

**Levels** (`levels=...`):

| Mode | Behaviour |
|---|---|
| `single` | one level, e.g. `levels.level=4-4` |
| `random` | each episode on a random level from a list |
| `curriculum` (default) | starts on 1-1 and unlocks the next level once the newest level and the average over all unlocked levels reach 50% success. Levels are sampled by failure rate, so forgotten levels get practiced again. |
| `progression` | the full game from 1-1, continuing after each flag. The episode ends at the first lost life. |
| `tuning` | for hyperparameter search: `curriculum` over 1-1, 1-3, 2-2 and 4-1, each with a different mechanic |

**Tracking.** Each training run is an [MLflow](https://mlflow.org) run in `mlflow.db`. It stores the resolved config
(`config.yaml`), the parameters, metrics per iteration (stepped by environment steps) and model checkpoints.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a CUDA GPU (the CPU works but is slow).

```sh
uv sync
uv run pytest                                    # tests
uv run ruff check && uv run ruff format --check  # lint and formatting, line length 120
uv run ty check                                  # types, for src/ and scripts/
```

## Configuration

Configs are composed with [Hydra](https://hydra.cc) from `configs/` and can be overridden on the command line:

| File | Contents |
|---|---|
| `neurio_config.yaml` | workers, iterations, steps per iteration, checkpoint frequency, curiosity, tracking URI |
| `agent/ppo.yaml` | PPO hyperparameters (gamma, GAE tau, clipping, loss weights, batch size, epochs) |
| `agent/model/*.yaml` | network: `impala` (default, residual encoder), `nature` (DQN encoder) |
| `agent/learning_rate/*.yaml` | `constant` (default), `linear_decay` |
| `agent/sampling_strategy/*.yaml` | `dist_sample` (default, sample from the policy), `greedy`, `adapt_eps_greedy` |
| `environment/*.yaml` | preprocessing and reward; `subsampled` (default) builds on `clipped` and `base` |
| `levels/*.yaml` | level modes, see above |

## Training with fixed parameters

```sh
uv run src/train.py                                    # defaults: curriculum over all levels, 5000 iterations
uv run src/train.py levels=single levels.level=1-3 num_iters=300 exploration.bonus_scale=0.5
uv run src/train.py agent.gamma=0.95 agent/learning_rate=linear_decay +tags.purpose=test
```

One iteration is 32 × 128 = 4096 environment steps and takes about 2.5–3.5 s on 32 cores and an RTX 4090.
Runs are grouped into MLflow experiments by level mode (e.g. `Neurio-lev-1-3`, `Neurio-lev-curriculum-32`).
`+tags.<name>=<value>` adds MLflow tags, for example to group runs.

Checkpoints are saved every `save_frequency` iterations and at the end. Curriculum runs also save a checkpoint when
the number of levels they clear improves; the run tag `best_checkpoint_iter` records it.

**Resuming.** Each checkpoint also stores the full training state: the network with its optimizer, the curriculum, the
self-imitation memory, the curiosity counts and the counters. Starting `train.py` with the same config as an
interrupted run asks whether to continue it from its last checkpoint or to start a new run. Tags, `render`,
`save_frequency`, `tracking_uri` and the test settings don't count as config. Without a terminal, e.g. with `nohup`,
set `resume=true` (continue, or start a new run if there is nothing to continue) or `resume=false` (always start a new
one). Work since the last checkpoint is repeated.

For long runs, keep the machine from suspending:

```sh
nohup systemd-inhibit --what=sleep:idle uv run src/train.py > train.log 2>&1 &
```

## Hyperparameter tuning

`src/tune.py` runs an [Optuna](https://optuna.org) study. Each trial is a normal training run, logged to MLflow.
Unpromising trials are stopped early based on the objective reported after each iteration.

```sh
uv run src/tune.py --study-name mechanics --search-space exploration --hours 10 --warmup-iters 250 -- \
    levels=tuning objective=mean_levels_cleared_score num_iters=700
```

- `--search-space` selects a search space defined in `src/tune.py`:
  - `ppo`: learning rate, gamma, GAE tau, critic and entropy weights, batch size and epochs.
  - `exploration`: self-imitation weight and memory size, curiosity bonus scale, gamma, entropy weight and learning
    rate.
- The objective is `recent_mean_reward` by default. With several levels, the mean reward mostly reflects which levels
  were played, so use `levels_cleared_score` (the expected number of levels cleared) or
  `mean_levels_cleared_score` (its mean over the run, which also rewards learning fast and steadily).
- `--n-trials` and/or `--hours` limit the study. The time limit only stops new trials; a running trial finishes.
- Overrides after `--` apply to every trial.
- `--pruner` chooses `median` (default), `hyperband` or `none`. `--warmup-iters` sets the number of iterations before
  a trial can be pruned (default 50).
- Studies are stored in `optuna.db`. Rerunning the same command resumes a study.
- Several processes can work on one study at the same time. Two parallel processes complete about 1.3× as many
  trials as one, since a single trial uses only about half of the CPU. Start them a few seconds apart, and give them
  different `--seed` values or none.

Tune on difficult levels with different mechanics, such as `levels=tuning`. 1-1 alone rewards running right and
favours short-sighted settings (gamma 0.9) that fail on later levels.

## Batches of experiments

`scripts/overnight.py` runs a fixed plan of training runs one after the other, each as its own process. Every run is
tagged with `experiment`, `variant`, `level` and `repeat`, and runs that would not finish before the deadline are
skipped. The plan is defined at the top of the script; currently it compares the DQN, a wider DQN and the
IMPALA encoder on 2-2, two runs at a time.

```sh
uv run scripts/overnight.py --dry-run                                            # print the plan
nohup systemd-inhibit --what=sleep:idle uv run scripts/overnight.py > overnight.log 2>&1 &
uv run scripts/overnight.py --report --experiment overnight-2026-10-04-1359      # summary per level and variant
```

The experiment name is printed when a batch starts. Each run's log is in `overnight/<experiment>/`.

## Watching a trained agent

`src/test.py` shows the agent playing one episode in a game window at the NES frame rate of 60 frames per second. By
default it plays the full game from 1-1 and continues after each flag until Mario loses a life, then logs where it
ended. The agent always takes its most likely action, so a checkpoint plays the same way every time. The model is
built from the run's own `config.yaml`, so it matches the checkpoint.

```sh
uv run src/test.py                                               # best curriculum run, latest checkpoint
uv run src/test.py test_run_id=<run id> test_iter=1189           # a specific run and checkpoint
uv run src/test.py test_run_id=<run id> levels=single levels.level=1-3   # a single level
```

- Without `test_run_id`, it picks the run with the best `objective` in the experiment of the configured level mode
  (by default `Neurio-lev-curriculum-32`; with `levels=single levels.level=1-3`, `Neurio-lev-1-3`).
- Without `test_iter`, it uses the latest checkpoint.
- Levels given on the command line replace the full game.
- Runs from before the config artifact was introduced cannot be loaded.

## Dashboards

```sh
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
uvx optuna-dashboard sqlite:///optuna.db                     # http://127.0.0.1:8080
```

Useful MLflow metrics:

| Metric | Meaning |
|---|---|
| `recent_mean_reward` | the training objective: mean episode reward of the last 100 episodes |
| `flag_rate`, `episode_x_pos_mean` / `_max` | share of episodes reaching the flag, distance reached |
| `num_unlocked_levels`, `success_rate_<level>`, `levels_cleared_score` | curriculum progress; the score is the expected number of levels cleared |
| `mean_levels_cleared_score` | the score's mean over the run so far |
| `novelty_bonus_per_step`, `visited_cells` | curiosity bonus and explored area |
| `actor_loss`, `critic_loss`, `entropy_loss`, `learning_rate` | PPO update |

## Project layout

| Path | Contents |
|---|---|
| `src/train.py` | `train()`, used by the command line and by `tune.py` |
| `src/tune.py`, `src/test.py` | hyperparameter search, evaluation |
| `src/environment.py` | environment and preprocessing wrappers, level selection |
| `src/game_loop.py` | interaction between environments and agent, metrics |
| `src/agents/` | PPO agent, experience buffer, GAE |
| `src/models/` | actor-critic network |
| `src/level_schedulers.py`, `src/novelty.py` | curriculum, curiosity bonus |
| `src/config/` | dataclasses behind the Hydra configs |
| `scripts/overnight.py` | batches of experiments |
| `test/` | tests; the PPO and model tests check the jaxtyping shape annotations at runtime |

## Findings so far

- **γ:** gamma 0.9 learns 1-1 about twice as fast as 0.99. On 1-3, though, no gamma, entropy or death-penalty setting
  got past the dead end at x = 783.
- **Death penalty:** a large one (300) makes the agent too cautious. It stalls until the timer runs out, since a
  penalty that far away is discounted to almost nothing.
- **Curiosity:** it helps, but not reliably. On 1-3, 3 of 11 runs with a bonus learned the level (scales 0.5 and 2.0),
  against 0 of 17 runs without one. Whether a run succeeds is decided in its first ~50 iterations, and the bonus
  becomes negligible soon after. On 4-4 the bonus roughly doubled the distance reached.
- **Self-imitation:** with curiosity, every run on 1-3 finds the route around the dead end, but few learn it. Imitating
  the 16 best own episodes per level (`agent.sil_loss_weight=1.0`) made all 8 of 8 runs learn it (2 of 8 without), and
  6 of 8 reached the flag within 100 iterations. The prioritized memory collapsed the policy instead (no flags).
- **Curriculum:** with forced unlocks after a timeout, the agent forgot earlier levels. Unlocking on mastery fixed
  that.
- **Curriculum tuning:** in 2M environment steps the best settings (self-imitation weight 0.6–0.8 with 4–6 episodes per
  level, bonus 0.6–0.85, gamma about 0.95) cleared world 1 and were still improving. None of them learned 2-1: one run
  died at the same spot (x ≈ 1760) in almost every episode, another still made progress up to x ≈ 2900.
- **2-1 alone:** every setting that ran 400 iterations on 2-1 learned it (flag in 44–86% of the final episodes), across
  gamma 0.90–0.98, self-imitation weights 0.1–2.6 and bonus scales 0.1–1.4. In the curriculum it only lacked practice.
- **Defaults:** the settings of the best curriculum trials (self-imitation 0.7 with 5 episodes per level, bonus 0.7,
  gamma 0.95, entropy 0.013, learning rate 4.5e-4).
- **Network:** in one calibration run each on 1-1, 1-3, 2-2 and 4-1, IMPALA cleared every level earlier than the DQN
  encoder (2-2 at iteration 641 against 717) at almost the same training time. Swimming (2-2) took about 500 iterations
  with both, against 50–100 for the other levels.
