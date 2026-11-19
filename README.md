# Neurio

A reinforcement learning agent that learns to play Super Mario Bros from the screen, with PPO or DreamerV3. The goal is
an agent that gets good at the game on its own: no human demonstrations and no level-specific scripting.

## Approach

**Environment.** [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) emulates the NES game through
[gymnasium](https://gymnasium.farama.org). 32 copies run in parallel worker processes. Each observation is processed as
follows:

- everything Mario cannot touch, such as the sky, hills, bushes and clouds, painted black (`environment.remove_background`).
  The game's RAM holds the grid of 16×16-pixel blocks in which it looks up collisions, and the emulator tells which
  pixels are sprites, e.g. Mario, enemies and moving platforms. Everything else is background.
- converted to grayscale
- cropped to 256×192 pixels, without the score bar and part of the ground
- resized to 64×48 pixels, a quarter

The full width stays: with 32 pixels cut on either side, the agent saw about 112 instead of 144 pixels ahead of Mario,
and on 5-3 it jumped onto narrow treetops whose Goombas were still right of its view, its most frequent death there.
- for PPO, stacked with the previous 3 frames, so that it can see motion

Each chosen action is held for 4 frames. The agent picks from the 12 button combinations of `COMPLEX_MOVEMENT`.

Each episode plays one level from its start and ends when Mario dies, reaches the flag or is cut off. Mario starts 1-1
small, like the game, and every later level as he left the previous one in game order: small, big or with the fire
flower, as often as the previous level's episodes reached the flag in each state during the last 1000 iterations
(`resume_frequency`, so that each period is saved complete; `powerups.StartPowerups`). Training thus practices the
starts that the agent's own play leads to, and evaluation, which carries Mario's state from level to level like the
game, meets the same ones. It starts with a random action for 0–3 frames, one less than an action is held, and
each frame keeps the previous frame's action with probability 0.25 (`environment.sticky_action_probability`), as in
DreamerV3's Atari setting ([Machado et al., 2018](https://arxiv.org/abs/1709.06009)). `test.py` plays without sticky
actions.
Otherwise every episode on a level would play out in exactly the same timing, and the agent could learn one exact course
that fails when it arrives a few frames off.

**Reward.** The agent learns from progress to the right (0.25 per pixel), reaching the flag (500) and dying (−50), all
scaled by 0.1 (`environment.progress_reward`, `flag_reward`, `death_penalty`, `reward_scale`). Changing the area through
a pipe or to the next level is not counted as progress. The area is the address of the part of the level the game has
loaded: the area number of gym-super-mario-bros stays the same in the rooms behind pipes, which made going down 1-1's
pipe look like a step back by about 900 pixels. With the game's own reward, which pays 1 per pixel and −25 for
dying, the agent learnt to take dead ends that pay more than they cost: on 1-3 it jumped into the pit before the tall
tree with two Goombas, since falling forward paid about 15 and dying cost 2.5. With dying at −100, DreamerV3 learnt to
clear 1-3, which it never did with the game's reward, but unlocked 1-2 and 1-3 at half the pace; −50 is in between.

Two stricter variants failed. Potential-based shaping with the progress as potential
([Ng et al., 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf))
makes the return from a state mostly minus the progress made so far, whatever the agent does next; Dreamer normalizes
advantages by the spread of returns, which then came mostly from how far Mario was, and learnt 1-1 several times
slower. Giving back all progress of the level on dying made deaths cost more the further Mario got, almost as much as
the flag at x ≈ 1150 of 1-3, while waiting cost nothing: in 40% of the episodes on 1-3, Mario stood on a platform until
the episode was truncated for lack of progress.

The flag pays 500 instead of the game's 50, as much as 50 pixels of progress, so that dead ends further to the right do
not pay more than finishing: in 1-2 the agent ran over the ceiling past the exit pipe into the warp zone and waited
there until the time ran out.

Leaving the level other than by its flag, i.e. through a warp pipe, ends the episode as not cleared, without reward or
penalty. The game would continue in the level the pipe leads to, whose progress and flag paid more than the level's own
flag: on 4-2, the agent took the warp zone to world 5 in every greedy episode, and 4-2 only counted as cleared with 5-1.

For PPO, the reward is divided by the running standard deviation of the discounted return,
as in Stable-Baselines3 and CleanRL. Without it, gamma 0.98 and the larger flag reward made the critic's loss about 6
times larger. It outweighed the policy in the shared network, the policy became almost deterministic, and the agent
ended up standing in front of the first pipe of 1-1 until the time ran out.

An episode is truncated after 150 steps without a new furthest position (`environment.max_steps_without_progress`;
entering another area counts as progress), so that dead ends do not fill the training with thousands of steps of
waiting. Truncated, not ended as a death: the agent learns from the value of the last state as if the episode went on,
so waiting costs what it costs in the game.

**Agents** (`agent=...`) all act on the same interface (`src/agents/agent.py`): `act` picks the next actions,
`observe` gets what they led to, and `update` learns after each iteration.

**PPO** (`agent=ppo`, default) with generalized advantage estimation (GAE) and gamma 0.98, so that the flag still counts
about 40 steps ahead, where the route to it is decided. The IMPALA residual encoder (3 stages with 32/64/64 channels and
a 256-unit layer) reads the frames; it generalizes better to unseen levels than the DQN encoder ("Nature CNN",
`agent/model=nature`). On the GPU the encoder runs in bfloat16. Together with an embedding of
the previous action, it feeds two outputs: action logits (the actor) and a state value (the critic). Each iteration
collects 128 steps per worker and trains 5 epochs on them. Advantages are normalized per minibatch. The entropy bonus
decays from 0.013 to 0.001 over the first 700 iterations (`agent.entropy_decay_updates`), as long as in the tuning runs.
Spread over a long run instead, it kept the policy so random that its own best episodes drifted out of reach.

**DreamerV3** (`agent=dreamer`, [Hafner et al.](https://danijar.com/project/dreamerv3/)), reimplemented in PyTorch
from the [official JAX code](https://github.com/danijar/dreamerv3), with its 12M parameter model and settings. It
learns a world model of the game from replayed sequences of its own play: a recurrent state that is inferred from the
frames, predicts the next state for each action, and from it the frame, the reward and whether the episode goes on.
The actor and critic learn only from rollouts of 15 steps that the world model imagines from replayed states, so the
agent can practise a jump many times without playing it. Returns are scaled by their 5th to 95th percentile range,
rewards and values are learned as distributions over exponentially spaced bins, so no reward scale needs tuning.

Two additions let it learn moves that rarely happen by chance, like the long jump onto the tall tree of 1-3, which needs
A held for about 7 steps. Without them, it died at the same pit for 4M steps:

- **Self-imitation:** an eighth of each batch (`agent.sil_fraction`) comes from its 8 best episodes per level (by game
  reward, `agent.sil_episodes_per_level`). The world model thereby learns a rare success from
  more than one sequence among a million, and the actor imitates the actions of these episodes where their return
  beats the critic.
- **Plan2Explore** ([Sekar et al., 2020](https://ramanans1.github.io/plan2explore/)): an ensemble of 8 heads predicts
  the next state from each state and action. Where they disagree, the world model has not learned yet what happens, and
  this disagreement is rewarded in imagination, so that the actor plans towards it. It has its own critic and return
  scale, normalized without the floor of 1 that keeps small differences in rare rewards from being magnified, and
  counts with a weight of 0.1 (`agent.exploration_weight`) against the reward.

Plan2Explore cannot find places the world model has never seen: in the maze of 4-4, the agent took the middle corridor
at the second fork for over 2,000 iterations, while only the bottom one leads on. Its entrance is a one-tile hole that
is only reached by walking back to the left from where Mario drops into the middle corridor. A **novelty bonus**
(`exploration.bonus_scale`, for any agent) therefore adds `1 / sqrt(visits)` to each step's reward, where visits counts
how often any worker has been in Mario's 16×16-pixel cell of the area; the world model learns it as part of the reward.
A new cell outweighs the progress lost by stepping left instead of right until about 10 visits. At 0.1, the agent came
within one cell of the hole but never stepped into it. Replacing up to 20% of the actions on a stalled level with
random ones did not help either: they never add up to walking back that far.

At 1.0, Mario found the hole about 25 times in 500 iterations, but mostly died: under it is open space over lava, and
he only lands on the bottom floor by letting go of left within two steps. The few safe landings were a handful of steps
among the million in the replay. Half of the replayed sequences (`agent.rare_fraction`) are therefore sampled around a
step picked with probability by its **rarity**, `1 / visits` of its cell when it was reached, so that rare experience is
replayed far more often than uniformly; without the novelty bonus, all steps are equally rare.

Neither helps with an action that is useless almost everywhere, e.g. pressing down: the actor generalizes across places
and rules it out everywhere, also where it has never been tried, and the world model confidently predicts that Mario
only crouches. On 8-4, the only way on is down a pipe floating above the ground. One run first pressed down on it about
21,000 iterations after unlocking 8-4, the next had not after 9,000. While training, a worker on a level that is
**stuck at a wall** therefore takes an **untried action** (`exploration.try_all_actions`) while there is one in Mario's
cell, the novelty bonus's, apart for Mario on the ground and in the air, like UCB1 plays every arm once. Which of them
comes first is sampled from the policy. A level is stuck once `levels.success_window` episodes in a row neither reached
the flag nor got further than before. The untried actions are only taken in a window of distance up to the wall, which
starts at the wall and moves back after every iteration that found nothing untried in it, also when no episode got
that far, until the way on is found, e.g. the pipe before 8-4's loop. Elsewhere the policy acts as usual: one run that
tried every action in every cell found the flags of new levels 3 to 10 times later, because at the frontier Mario
hardly got further. The actor still
learns in imagination, from the world model, which learns from every action taken; CMPO then values pressing down on the
pipe like any other action. PPO learns on-policy and does without it.

Differences to the original:

- The replay keeps the last 1M steps instead of 5M, about 6.5 GB with the model state of each step.
- Plan2Explore's reward is added to the task reward of one actor, instead of training a separate exploration actor.
- The actor is also regularized towards Muesli's clipped MPO policy ([Hessel et al., 2021](https://arxiv.org/abs/2104.06159)):
  its policy times exp(advantage), with the advantages of all actions of each replayed state valued one imagined step
  ahead (reward plus critic value), normalized like the returns and clipped to ±1 (`agent.loss_scales.cmpo`), so that
  it also learns of better actions it rarely samples. Unlike Muesli, the prior is the current policy instead of a
  target network, the target is label smoothed by 3e-5 (`agent.cmpo_label_smoothing`), and the advantages include
  Plan2Explore's like those of the policy gradient: the actor also tries actions whose outcome the world model is
  unsure about. On 8-4 the agent learnt to jump onto the ledge of the second right pipe, but never pressed down there:
  the novelty bonus of the ledge wore off with every visit, while the disagreement about pressing down only goes once
  it has been tried. It costs about 18% of an update.

  Both work on the actor's own policy, before the 1% uniform mixture with which actions are sampled. That policy had
  become deterministic underneath the mixture: on 8-4 it pressed down on the first pipe with a probability of 1e-28,
  while the world model valued that action at 29–123 against 21 for every other. The entropy bonus, like CMPO and the
  policy gradient, changes an action's probability in proportion to it, so nothing could raise it again; label
  smoothing pulls by a constant smoothing / 12 (the difference between label smoothing and a confidence penalty,
  [Pereyra et al., 2017](https://arxiv.org/abs/1701.06548)). A first version took the acting policy, with its uniform
  mixture, as the prior: in effect label smoothing by 1%, which made the policy almost random within 1000 updates.
  Smoothing by 3e-4 raised the probability of down on 8-4's pipe from 1e-28 within a few hundred iterations, so that
  the agent took the pipe, but also doubled the entropy: where actions are of about equal value, the pull towards
  uniform is all that moves the policy, and the optimizer, which normalizes each weight's updates, moves it fast however
  weak it is. In about a tenth of the states 10-20% of the probability spread over other actions.

The updates owed to each step run while the workers play it, as in the original. With the default 32 replays per
step, that is one update of 16 sequences of 64 steps per step of the 32 workers, and an iteration of 128 steps takes
about 10.5 s on an RTX 4090 at 48×48 frames (64×48 makes an update about 9% slower), after compiling the networks for about 2.5 minutes at the start. The
updates take about 8 s of it. Fewer than one update per step overlaps worse: with 24 workers, 512 steps per iteration
and 10.67 replays per step, the same 128 updates took 15.5 s, since the next action waits for each update.

**Self-imitation** (PPO; DreamerV3 above). `agent.sil_loss_weight` (default 0.7, 0 disables it): the agent keeps its
own 5 best episodes per level (by game reward, `agent.sil_episodes_per_level`) and, after each update, imitates their
actions wherever their return beats its current value estimate, so that it learns a better route it found by chance
before it commits to a worse one.

**Levels** (`levels=...`):

| Mode | Behaviour |
|---|---|
| `single` | one level, e.g. `levels.level=4-4` |
| `random` | each episode on a random level from a list |
| `curriculum` (default) | starts on 1-1 and unlocks the next level once the agent reached the flag on the newest level and the average success over all unlocked levels is 50%. Levels are sampled by failure rate, so forgotten levels get practiced again. |
| `tuning` | for hyperparameter search: `curriculum` over 1-1, 1-3, 2-2 and 4-1, each with a different mechanic |

**Tracking.** Each training run is an [MLflow](https://mlflow.org) run in `mlflow.db`. It stores the resolved config
(`config.yaml`), the parameters, metrics per iteration (see [Dashboards](#dashboards)) and model checkpoints.

## Setup

Requires [uv](https://docs.astral.sh/uv/), a C++ compiler and a CUDA GPU (the CPU works but is slow). The NES
emulator is a fork of [nes-py](https://github.com/Kautenja/nes-py) in `external/nes-py` that also reports which layer
each pixel shows; uv builds it.

```sh
git submodule update --init
uv sync
uv run pytest                                    # tests
uv run ruff check && uv run ruff format --check  # lint and formatting, line length 120
uv run ty check                                  # types, for src/ and scripts/
```

## Configuration

Configs are composed with [Hydra](https://hydra.cc) from `configs/` and can be overridden on the command line:

| File | Contents |
|---|---|
| `neurio_config.yaml` | workers, iterations, steps per iteration, checkpoint frequency, tracking URI |
| `agent/ppo.yaml` | PPO hyperparameters (gamma, GAE tau, clipping, loss weights, batch size, epochs) |
| `agent/model/*.yaml` | PPO's network: `impala` (default, residual encoder), `nature` (DQN encoder) |
| `agent/learning_rate/*.yaml` | PPO's learning rate: `constant` (default), `linear_decay` |
| `agent/dreamer.yaml` | DreamerV3 network sizes and training settings |
| `agent/random.yaml` | random actions, as a baseline |
| `environment/base.yaml` | preprocessing and reward |
| `levels/*.yaml` | level modes, see above |

## Training with fixed parameters

```sh
uv run src/train.py                                    # defaults: curriculum over all levels, 5000 iterations
uv run src/train.py levels=single levels.level=1-3 num_iters=300 agent=dreamer
uv run src/train.py agent.gamma=0.95 agent/learning_rate=linear_decay +tags.purpose=test
uv run src/train.py agent=dreamer levels=single levels.level=1-1
```

One iteration is 32 × 128 = 4096 environment steps and takes about 2.5–3.5 s with PPO on 32 cores and an RTX 4090.
Runs are grouped into MLflow experiments by level mode (e.g. `Neurio-lev-1-3`, `Neurio-lev-curriculum-32`).
`+tags.<name>=<value>` adds MLflow tags, for example to group runs.

Checkpoints are saved every `save_frequency` iterations and at the end. Curriculum runs also save a checkpoint when
the number of levels they clear improves; the run tag `best_checkpoint_iter` records it.

**Resuming.** Every `resume_frequency` (1000) iterations and after the last one, a run stores the full training state:
the networks with their optimizer, Dreamer's replay and best episodes, PPO's self-imitation memory, the curriculum, the
novelty counts and the counters, about 8 GB for Dreamer. So a resumed run continues as if it had not been interrupted,
instead of relearning from an empty replay. The state alternates between two files in the run's artifacts, so that an
interrupted save leaves the previous one intact. Starting `train.py` with the same config as an interrupted run asks
whether to continue it from its last saved state or to start a new run. Tags, `render`, `save_frequency`,
`resume_frequency`, `tracking_uri`, the test settings and `num_iters` don't count as config, so a finished run can be
continued with a higher `num_iters`. Without a terminal, e.g. with `nohup`, set `resume=true` (continue, start a new run
if there is none to continue, or exit if the run already reached `num_iters`) or `resume=false` (always start a new
one). Work since the last saved state is repeated: resuming first deletes the checkpoints and metrics the run logged
after that state, also those of greedy evaluations, since they belong to the discarded course of training
(`resume.discard_after`). This needs the tracking database, as MLflow has no API for deleting metrics.

For long runs, keep the machine from suspending:

```sh
nohup systemd-inhibit --what=sleep:idle uv run src/train.py > train.log 2>&1 &
```

## Hyperparameter tuning

`src/tune.py` runs an [Optuna](https://optuna.org) study. Each trial is a normal training run, logged to MLflow.
Unpromising trials are stopped early based on the objective reported after each iteration.

```sh
uv run src/tune.py --study-name mechanics --search-space exploration --hours 10 --warmup-iters 250 -- \
    levels=tuning objective=cleared_mean num_iters=700
```

- `--search-space` selects a search space defined in `src/tune.py`:
  - `ppo`: how PPO learns from its experience: learning rate, epochs, batch size, GAE tau, clip range and critic
    weight.
  - `exploration`: self-imitation weight and memory size, gamma, entropy weight and learning
    rate.
- The objective is `cleared` by default, the expected number of levels cleared from the recent success rates. Use
  `cleared_mean`, its mean over the run, to also reward learning fast and steadily.
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

`src/test.py` shows the agent playing in a game window at the NES frame rate of 60 frames per second. It plays the
run's levels in order, e.g. 1-1, 1-2, … for the curriculum, from small Mario in 1-1 on and with the power-up he
reached each flag with, like in the game, and stops at the first level it does not clear. It logs where each episode ended. The agent always takes its most likely
action and the episodes are seeded, so a checkpoint plays the same way every time. The model is built from the run's own `config.yaml`, so it
matches the checkpoint.

```sh
uv run src/test.py                                               # best curriculum run, latest checkpoint
uv run src/test.py test_run_id=<run id> test_iter=1189           # a specific run and checkpoint
uv run src/test.py test_run_id=<run id> levels=single levels.level=1-3   # a single level
```

- Without `test_run_id`, it picks the run with the best `objective` in the experiment of the configured level mode
  (by default `Neurio-lev-curriculum-32`; with `levels=single levels.level=1-3`, `Neurio-lev-1-3`).
- Without `test_iter`, it uses the latest checkpoint.
- Levels given on the command line replace the run's own.
- Runs from before the config artifact was introduced cannot be loaded.

`src/greedy_progress.py` plays this greedy run, without a window, for every checkpoint of a run (selected the same
way). Each checkpoint's progress is the sum of the distances its episodes got: to the flags of the cleared levels plus
to where it failed. The distance counts the changes of x within each area, as the reward does, so it does not drop
when Mario goes down a pipe into a room further on, whose x positions start again from the left; the end x position
did, e.g. in 8-4's water stretch. It logs `greedy/progress` and `greedy/levels_cleared` to the run with the checkpoint's
iteration as step, tags the earliest best checkpoint as `best_greedy_iter` (for `test_iter`) and plots the progress
to `plots/` and the run's artifacts. Calling it again only evaluates new checkpoints. Each one takes about 30–60 s.

```sh
uv run src/greedy_progress.py test_run_id=<run id>
```

Checkpoints take about 72 MB each. `src/clean_checkpoints.py` deletes those of a run that were evaluated, except each
one that cleared more levels than every checkpoint before it, the ones tagged `best_checkpoint_iter` and
`best_greedy_iter`. Checkpoints not evaluated yet are kept. Resuming doesn't need checkpoints.

```sh
uv run src/clean_checkpoints.py <run id>            # list what would be deleted
uv run src/clean_checkpoints.py <run id> --apply    # delete it
```

## World model previews

For DreamerV3 runs, each checkpoint also stores `preview.gif`: replayed sequences of the game, of which the world model
sees the first quarter and then predicts the rest from the actions taken. Each column shows the true frames, the
predicted ones and their difference (gray where they agree); the border turns from green to red where the prediction
starts. Small, fast things like Mario, enemies and fireballs fade first when the model is unsure.

`src/preview.py` renders the same for any checkpoint and level, from episodes it plays itself, plus a "dream": the
same start, continued with actions of the agent's own policy and no real frames. It selects the run like `test.py`
and writes the GIFs to `previews/`:

```sh
uv run src/preview.py test_run_id=<run id> levels=single levels.level=1-2
```

## Dashboards

```sh
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
uvx optuna-dashboard sqlite:///optuna.db                     # http://127.0.0.1:8080
```

Metrics are logged once per iteration, with the iteration as MLflow's step. A **step** is one action of every worker,
held for 4 frames; an **iteration** is 128 steps of all 32 workers followed by the agent's update; an **episode** is one
attempt at a level, until Mario dies, reaches the flag or is cut off.

| Metric | Meaning |
|---|---|
| `levels/success_<level>`, `levels/distance_<level>` | share of the level's last 50 episodes that reached the flag, and the mean distance they got (see `greedy/progress`) |
| `levels/cleared`, `levels/cleared_mean` | the expected number of levels cleared (sum of the success rates), and its mean over the run so far |
| `levels/unlocked` | curriculum: number of unlocked levels |
| `world_model/reconstruction`, `/reward`, `/dynamics` | DreamerV3: how well the world model predicts the frames, the rewards and its own next state |
| `behavior/entropy` | randomness of the policy, between 0 and ln 12 ≈ 2.5 |
| `behavior/cmpo_changes` | DreamerV3: share of replayed states where the CMPO target makes another action the most likely |
| `behavior/imagined_return`, `/return_scale` | DreamerV3: the return the actor expects in imagination, and the spread of returns that advantages are divided by |
| `exploration/plan2explore_reward` | Plan2Explore's reward per imagined step |
| `powerups/start_<level>` | share of the level's episodes that start with a power-up, from the previous level's finishes |
| `exploration/novelty_bonus`, `/visited_cells` | the novelty bonus per step, and the number of cells visited so far |
| `exploration/untried_actions` | share of the steps that took an action untried in Mario's cell |
| `exploration/stuck_levels`, `/window_<level>` | levels stuck at a wall, and per level the window before the wall in which untried actions are taken, in pixels (0 while not stuck) |
| `progress/env_steps`, `/updates` | steps of all workers so far, and DreamerV3's updates so far |
| `greedy/progress`, `/levels_cleared` | per checkpoint, logged by `greedy_progress.py`: progress and levels cleared in one greedy go from 1-1 |

## Project layout

| Path | Contents |
|---|---|
| `src/train.py` | `train()`, used by the command line and by `tune.py` |
| `src/tune.py` | hyperparameter search |
| `src/test.py`, `src/greedy_progress.py`, `src/clean_checkpoints.py`, `src/preview.py`, `src/evaluation.py` | watching a trained agent, greedy progress of all checkpoints and deleting the uninteresting ones, world model previews, loading and playing them |
| `src/environment.py` | environment and preprocessing wrappers, level selection |
| `src/game_loop.py`, `src/exploration.py` | interaction between environments and agent, metrics; the novelty bonus |
| `src/agents/` | the agent interface and the random agent |
| `src/agents/ppo/` | PPO: agent, actor-critic networks, GAE, self-imitation, reward scaling |
| `src/agents/dreamer/` | DreamerV3: agent, world model, networks, replay, best episodes, optimizer |
| `src/level_schedulers.py`, `src/powerups.py` | curriculum, recent results per level; the levels' start power-ups |
| `src/config/` | dataclasses behind the Hydra configs |
| `scripts/overnight.py` | batches of experiments |
| `test/` | tests; the PPO and model tests check the jaxtyping shape annotations at runtime |

## Findings so far

- **γ:** gamma 0.9 learns 1-1 about twice as fast as 0.99. On 1-3, though, no gamma, entropy or death-penalty setting
  got past the dead end at x = 783.
- **Death penalty:** a large one (300) makes the agent too cautious. It stalls until the timer runs out, since a
  penalty that far away is discounted to almost nothing.
- **Curiosity** (a count-based bonus for reaching rarely visited positions, removed since): it helps, but not reliably. On 1-3, 3 of 11 runs with a bonus learned the level (scales 0.5 and 2.0),
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
- **Mechanics tuning** (22 trials on 1-1, 1-3, 2-2 and 4-1, 700 iterations): the best trials scored as well as the
  defaults (mean levels cleared 1.51 against 1.50). Gamma above 0.96, a curiosity bonus below 0.3 and an initial
  entropy weight below 0.008 were clearly worse; the self-imitation weight (0.4–1.9), its memory size (6–51 episodes)
  and the learning rate (1e-4–1e-3) hardly mattered. 2-2 stayed the bottleneck: the best runs reached its flag in 30–40%
  of the episodes, dying to enemies all over the second half of the level rather than at one obstacle.
