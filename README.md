# Neurio

A reinforcement learning agent that learns to play all 32 levels of Super Mario Bros from the screen, with DreamerV3 or,
as a baseline, PPO. The agent learns on its own: no human demonstrations, no restarts from saved states and no
level-specific scripting.

## Environment

[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) emulates the NES game through
[gymnasium](https://gymnasium.farama.org); 32 copies run in parallel worker processes (`src/environment.py`).

**Observations.** Each frame is processed as follows:

- Everything Mario cannot touch, such as the sky, hills, bushes and clouds, is painted black (`remove_background`). The
  game's RAM holds the grid of 16×16-pixel blocks in which it looks up collisions, and the patched emulator tells which
  pixels are sprites, e.g. Mario, enemies and moving platforms. The agent thus sees what matters for
  playing, the same way on every level.
- Converted to grayscale and cropped to 256×192 pixels, without the status bar and part of the ground. The full width
  stays: with 32 pixels cut on either side, the agent jumped onto 5-3's narrow treetops without seeing the Goombas on
  them.
- Resized to 64×48 pixels, a quarter. PPO, which has no memory, sees the last 4 frames to perceive motion.

**Actions.** The agent picks from the 12 button combinations of `COMPLEX_MOVEMENT`; each is held for 4 frames. Each frame
keeps the previous frame's action with probability 0.25 (sticky actions, as in DreamerV3's Atari setting,
[Machado et al., 2018](https://arxiv.org/abs/1709.06009)), and each episode starts with a random action for 0–3 frames.
Otherwise every episode on a level plays out in exactly the same timing, and the agent could learn one exact course that
fails when it arrives a few frames off. Evaluation plays without sticky actions.

**Episodes.** Each episode plays one level from its start until Mario dies, reaches the flag, leaves the level or is cut
off. Mario starts 1-1 small, like the game, and every later level as he left the previous one: small, big or with the
fire flower, as often as the previous level's episodes reached its flag in each state during the last 1000 iterations
(`powerups.StartPowerups`). Training thus practises the starts that the agent's own play leads to, and evaluation, which
carries Mario's state from level to level like the game, meets the same ones.

**Reward** (`environment.ProgressShaping`). The agent learns from progress to the right (0.25 per pixel), reaching the
flag (500) and dying (−50), all scaled by 0.1. With the game's own reward (1 per pixel, −25 for dying, each frame's clipped to ±15), dead ends paid
more than they cost: on 1-3 the agent jumped into a pit, since falling forward paid more than dying cost. The flag pays
as much as 2000 pixels of progress, so that no dead end further right pays more than finishing, e.g. waiting in 1-2's
warp zone until the time runs out.

- Changing the area, e.g. through a pipe, is not counted as progress. The area is the address of the part of the level
  the game has loaded; gym-super-mario-bros's area number stays the same in the rooms behind pipes, which made going
  down 1-1's pipe look like a step back by 900 pixels.
- Leaving the level other than by its flag, i.e. through a warp pipe, ends the episode as not cleared, without reward or
  penalty. Otherwise the levels the pipe leads to pay more than the level's own flag, and on 4-2 the agent took the warp
  zone in every greedy episode.
- An episode is truncated after 150 steps without getting further right, so that dead ends do not fill the training
  with waiting. Truncated, not ended: the agent learns from the value of the last state as if the episode went on.

The game's own reward, which also pays for score, coins and power-ups, only ranks the agent's best episodes for
self-imitation, with the flag counted as 500.

## Exploration

Three mechanisms help the agent find what it would rarely find by chance (`src/exploration.py`). The replay of rare
steps is DreamerV3's; both agents get the others.

**Novelty bonus.** Each step's reward gets `1 / sqrt(visits)`, where visits counts how often any worker has been in
Mario's 16×16-pixel cell of the area. A new cell outweighs the progress lost by stepping left instead of right until
about 10 visits. In the maze of 4-4 the way on is a one-tile hole that is only reached by walking back to the left;
with a bonus of 0.1 the agent came within a cell of it but never stepped in, with 1.0 it found it.

**Replay of rare steps.** Half of DreamerV3's replayed sequences are sampled around a step picked by its rarity,
`1 / visits` of its cell when it was reached; an episode's first frame has the rarity of its start, which is among the
most visited cells. The few safe landings through 4-4's hole were otherwise a handful of steps among a million in the
replay.

**Untried actions at walls.** The actor generalizes across places: an action that is useless almost everywhere, like
pressing down, becomes unlikely also where it has never been tried. On 8-4, the only way on is down a pipe floating
above the ground, and a run took 21,000 iterations after unlocking 8-4 to press down on it. A level is stuck at a wall
once 50 episodes in a row (`levels.success_window`) neither reached the flag nor got a cell further than any before.
Then a worker within a window of distance before the wall takes an action that has never been taken in its cell, on the
ground or in the air, like UCB1 plays every arm once; the policy decides their order. The window starts at the wall and
moves back a cell after every iteration that found nothing untried in it, until the way on is found, e.g. the pipe
before 8-4's loop. Progress past the wall closes it. Elsewhere the policy acts as usual: taking every untried action in
every new cell made the agent find the flags of new levels 3 to 10 times later, since at the frontier Mario hardly got
further.

## Curriculum

The levels are unlocked in game order (`levels=curriculum`, `src/level_schedulers.py`). The next level is unlocked once
the newest one reached its flag in any of its last 50 episodes, and the success rate averaged over all unlocked levels
is at least 50%. A newly reached level then becomes reliable while the next ones are already played: waiting for 50% on
each new level spent about 27,000 of the 38,000 iterations until 8-4 was unlocked on these climbs, one after the other.
The average keeps the pool from filling up with levels that are not learned yet; unlocking all 32 levels at once learned
far slower than the curriculum. Episodes are sampled by need: each level's weight is its failure rate, at least 0.1, so
the agent mostly practises what it cannot do yet, including what it forgot.

## Agents

All agents act on the same interface (`src/agents/agent.py`): `act` picks the next actions, `observe` gets what they led
to, and `update` learns after each iteration.

### DreamerV3

`agent=dreamer` (default), [Hafner et al., 2023](https://danijar.com/project/dreamerv3/), reimplemented in PyTorch from
the [official JAX code](https://github.com/danijar/dreamerv3) with its 12M parameter model and settings
(`src/agents/dreamer/`). It learns a world model from replayed sequences of its own play: a recurrent state, inferred
from the frames, that predicts the next state for each action and from it the frame, the reward and whether the episode
goes on. The actor and critic learn from 15-step rollouts that the world model imagines from replayed states, so the
agent can practise a jump many times without playing it.

Additions, for moves that rarely happen by chance. Without the first two, the agent died at 1-3's tall tree, which needs
the jump button held for about 7 steps, for 4M steps:

- **Self-imitation** ([Oh et al., 2018](https://arxiv.org/abs/1806.05635)): an eighth of each batch comes from the
  agent's 8 best episodes per level and start power-up. The world model learns a rare success from more than one
  sequence among a million, and the actor imitates its actions where their return beats the critic. The episodes keep
  the reward without the novelty bonus, which wears off, so that a success stays worth what it is to the task.
- **Plan2Explore** ([Sekar et al., 2020](https://ramanans1.github.io/plan2explore/)): an ensemble of 8 heads predicts the
  next state from each state and action. Their disagreement, where the world model has not learned yet what happens, is
  rewarded in imagination with a weight of 0.1 against the task, so that the actor plans towards it. It is added to the
  one actor's advantages instead of training a separate exploration actor.
- **CMPO** ([Hessel et al., 2021](https://arxiv.org/abs/2104.06159)): the actor is also pulled towards its policy times
  exp(advantage), with the advantages of all actions of each replayed state valued one imagined step ahead and clipped
  to ±1, so that it learns of better actions it rarely samples. The prior is the current policy instead of Muesli's
  target network. The target is label smoothed by 3e-5: the policy underneath the 1% uniform mixture with which actions
  are sampled had become deterministic, pressing down on 8-4's pipe with a probability of 1e-28, and every other loss
  changes an action's probability in proportion to it. Smoothing by 3e-4 doubled the entropy.

The replay keeps the last 1M steps instead of 5M. Each replayed sequence continues from the model state of the step
before it, as stored; training writes back the states it computes for the replayed steps, so that the stored ones stay
those of the current model. The best episodes may be kept for the whole run, and their states would otherwise be those
of a model from tens of thousands of updates before. The updates owed to each step run while the workers play it: with 32
replays per step, one update of 16 sequences of 64 steps per step of the 32 workers. An iteration of 128 steps takes
about 11.4 s on an RTX 4090, after compiling the networks for a few minutes at the start.

### PPO

`agent=ppo`, a baseline ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347), `src/agents/ppo/`) with
generalized advantage estimation and gamma 0.98, so that the flag still counts about 40 steps ahead. The IMPALA residual
encoder ([Espeholt et al., 2018](https://arxiv.org/abs/1802.01561)), twice as wide as the original, reads the stacked
frames; together with an embedding of the previous action it feeds the action logits and the state value. Each
iteration trains 5 epochs on the 128 steps of each worker.

- The reward is divided by the running standard deviation of the discounted return, as in Stable-Baselines3 and
  CleanRL. Without it, the critic's loss outweighed the policy in the shared network, and the agent ended up standing
  in front of 1-1's first pipe.
- The entropy bonus decays from 0.013 to 0.001 over the first 700 updates. Spread over a long run instead, it kept the
  policy so random that its own best episodes drifted out of reach.
- Self-imitation: the agent keeps its 5 best episodes per level and start power-up and after each update imitates
  their actions wherever their return beats its value estimate, with the reward without the novelty bonus.
- Like DreamerV3, it samples actions from its policy mixed with 1% of the uniform distribution, and takes untried
  actions at walls. These are off-policy for PPO: it learns from them with its policy's probability of them, which its
  clipping bounds, while self-imitation learns from an episode that got through without that restriction.

## Results

Both agents trained on the curriculum for 30,000 iterations (123M steps) on an RTX 4090. PPO's iterations are faster,
so it trained for 28.6 hours to DreamerV3's 95.8:

| | DreamerV3, 28.6 h (iteration 8,991) | DreamerV3, 30,000 iterations | PPO, 30,000 iterations |
|---|--:|--:|--:|
| Levels unlocked | 28 | 32 | 9 |
| Sum of training success rates | 13.2 | 23.6 | 3.6 |
| Most levels cleared in one greedy go, up to then | 11 | 32 | 8 |

The sums of success rates are means over the last 250 iterations up to that point.

### DreamerV3


- All 32 levels were unlocked at iteration 12,914 (41.1 h). 8-4 took by far the longest to its first flag, 3,482
  iterations: its way on is down a pipe that the agent had to try first, see untried actions above. No other level took
  more than 857.
- The first greedy go through all 32 levels, from small Mario in 1-1 to 8-4's flag, came at checkpoint 27,300, after
  86.9 hours of training; 27,500 did it too. A greedy go is fragile: the checkpoints from 27,000 to 30,000 cleared a
  median of 8 levels in a row.
- The final checkpoint's success rates per level, measured with `sampled_success.py` (1024 episodes per level, from
  small Mario, standard error at most 1.6 percentage points), average 79%. Its weakest level is 5-3 at 32%, whose
  Bullet Bills fire at random.

| Level | Unlocked at | First flag at | Iterations to first flag | Success |
|---|--:|--:|--:|--:|
| 1-1 | 0 | 125 | 125 (0.4 h) | 94% |
| 1-2 | 415 | 673 | 258 (0.8 h) | 88% |
| 1-3 | 1,009 | 1,108 | 99 (0.3 h) | 85% |
| 1-4 | 1,248 | 1,375 | 127 (0.4 h) | 83% |
| 2-1 | 1,614 | 1,653 | 39 (0.1 h) | 86% |
| 2-2 | 1,893 | 2,233 | 340 (1.1 h) | 70% |
| 2-3 | 2,234 | 2,269 | 35 (0.1 h) | 87% |
| 2-4 | 2,593 | 2,646 | 53 (0.2 h) | 87% |
| 3-1 | 2,790 | 3,136 | 346 (1.1 h) | 87% |
| 3-2 | 3,169 | 3,188 | 19 (0.1 h) | 94% |
| 3-3 | 3,393 | 3,773 | 380 (1.2 h) | 85% |
| 3-4 | 3,774 | 3,925 | 151 (0.5 h) | 84% |
| 4-1 | 3,927 | 3,959 | 32 (0.1 h) | 90% |
| 4-2 | 4,005 | 4,430 | 425 (1.4 h) | 88% |
| 4-3 | 4,431 | 4,627 | 196 (0.6 h) | 82% |
| 4-4 | 4,733 | 5,336 | 603 (2.0 h) | 90% |
| 5-1 | 5,337 | 5,414 | 77 (0.3 h) | 89% |
| 5-2 | 5,415 | 5,838 | 423 (1.4 h) | 80% |
| 5-3 | 5,839 | 6,441 | 602 (2.0 h) | 32% |
| 5-4 | 6,443 | 6,598 | 155 (0.5 h) | 73% |
| 6-1 | 6,600 | 6,682 | 82 (0.3 h) | 90% |
| 6-2 | 6,683 | 6,951 | 268 (0.8 h) | 80% |
| 6-3 | 6,976 | 7,529 | 553 (1.7 h) | 77% |
| 6-4 | 7,539 | 7,823 | 284 (0.9 h) | 77% |
| 7-1 | 7,825 | 7,926 | 101 (0.3 h) | 67% |
| 7-2 | 8,386 | 8,405 | 19 (0.1 h) | 57% |
| 7-3 | 8,433 | 8,435 | 2 (0.0 h) | 85% |
| 7-4 | 8,842 | 9,425 | 583 (1.8 h) | 79% |
| 8-1 | 9,807 | 10,274 | 467 (1.5 h) | 76% |
| 8-2 | 10,515 | 11,372 | 857 (2.7 h) | 71% |
| 8-3 | 11,374 | 11,523 | 149 (0.5 h) | 60% |
| 8-4 | 12,914 | 16,396 | 3,482 (11.1 h) | 54% |

Iterations are counted from the start of training, those to the first flag and their hours of training from the
level's unlock. The run was interrupted four times and resumed from its last saved state, repeating the iterations
since; times count each iteration once.

The run used the code of commit cc791f4. Since then, four changes affect training: an episode's first frame gets the
rarity of its start cell instead of 1, which had made episode starts 81% of the rarity the replay samples by; the best
episodes keep the reward without the novelty bonus; they are ranked by the game's reward without its clipping of each
frame to ±15; and training writes back the model states it computes.

### PPO

PPO found the flag of each level it unlocked within 821 iterations, but its success rates on the unlocked levels stayed
too low for the curriculum's mean of 0.5 to open the next one. The unlocks came further and further apart; the last,
3-1, came at iteration 17,516, and the sum of its training success rates stayed at about 3.6 from there to the end.
Its best greedy go, at checkpoint 27,300, cleared the first 8 levels. The final checkpoint's sampled success rates
average 49% on its 9 levels and at most 1.8% on each of the 23 it never trained on.

| Level | Unlocked at | First flag at | Iterations to first flag | Success |
|---|--:|--:|--:|--:|
| 1-1 | 0 | 14 | 14 (0.0 h) | 85% |
| 1-2 | 97 | 134 | 37 (0.0 h) | 63% |
| 1-3 | 422 | 643 | 221 (0.2 h) | 55% |
| 1-4 | 761 | 815 | 54 (0.1 h) | 24% |
| 2-1 | 1,015 | 1,345 | 330 (0.3 h) | 55% |
| 2-2 | 2,560 | 3,381 | 821 (0.8 h) | 19% |
| 2-3 | 6,467 | 6,918 | 451 (0.4 h) | 51% |
| 2-4 | 12,020 | 12,118 | 98 (0.1 h) | 65% |
| 3-1 | 17,516 | 18,074 | 558 (0.5 h) | 24% |

The run used `agent=ppo num_iters=30000` with the code described in this README. It was stopped three times and
resumed, counted as for DreamerV3.

## Setup

Requires [uv](https://docs.astral.sh/uv/), a C++ compiler and a CUDA GPU (the CPU works but is slow). The NES emulator
is [nes-py](https://github.com/Kautenja/nes-py), pinned as a submodule in `external/nes-py`, with a patch that tells
for each screen pixel whether it shows the backdrop, the background or a sprite (`patches/nes-py-layer-buffer.patch`,
for `environment.remove_background`). uv builds it.

```sh
git submodule update --init
git -C external/nes-py apply ../../patches/nes-py-layer-buffer.patch
uv sync
uv run pytest                                    # tests
uv run ruff check && uv run ruff format --check  # lint and formatting, line length 120
uv run ty check                                  # types
```

## Configuration

Configs are composed with [Hydra](https://hydra.cc) from `configs/` and can be overridden on the command line:

| File | Contents |
|---|---|
| `neurio_config.yaml` | workers, iterations, steps per iteration, checkpoint and resume frequency, tracking URI, novelty bonus |
| `agent/dreamer.yaml` | DreamerV3 network sizes and training settings |
| `agent/ppo.yaml` | PPO network and training settings |
| `environment/base.yaml` | preprocessing and reward |
| `levels/curriculum.yaml`, `levels/single.yaml` | the curriculum over all levels, or one level, e.g. `levels=single levels.level=4-4` |

## Training

```sh
uv run src/train.py                                         # DreamerV3, curriculum over all levels
uv run src/train.py agent=ppo num_iters=30000
uv run src/train.py levels=single levels.level=1-3 num_iters=300 +tags.purpose=test
```

Each run is an [MLflow](https://mlflow.org) run in `mlflow.db`, grouped into experiments by level mode (e.g.
`Neurio-lev-curriculum-32`). It stores the resolved config (`config.yaml`), from which evaluation rebuilds the agent,
the parameters, the metrics of each iteration and the checkpoints. Checkpoints are saved every `save_frequency`
iterations, at the end, and when the number of levels cleared improved (tag `best_checkpoint_iter`).
`+tags.<name>=<value>` adds MLflow tags.

**Resuming.** Every `resume_frequency` iterations and after the last one, a run stores the full state of training: the
networks with their optimizer, DreamerV3's replay and best episodes, PPO's self-imitation memory, the curriculum, the
exploration statistics and the counters, about 8 GB for DreamerV3. A resumed run continues as if it had not been
interrupted. The state alternates between two files, so that an interrupted save leaves the previous one intact.
Starting `train.py` with the same config as an interrupted run asks whether to continue it; tags, the frequencies, the
tracking URI, the test settings and `num_iters` don't count as config, so a finished run can be continued with a higher
`num_iters`. Without a terminal, set `resume=true` or `resume=false`. Resuming deletes what the run logged after its
saved state, checkpoints and metrics, since the resumed run repeats those iterations (`resume.discard_after`).

For long runs, keep the machine from suspending:

```sh
nohup systemd-inhibit --what=sleep:idle uv run src/train.py > train.log 2>&1 &
```

## Evaluation

The agent always takes its most likely action and the episodes are seeded, so a checkpoint plays the same way every
time. It plays the run's levels in order from small Mario in 1-1 on, with the power-up he reached each flag with, like
the game, and stops at the first level it does not clear. Without `test_run_id`, the run that cleared the most levels in
the experiment of the configured level mode is used; without `test_iter`, its latest checkpoint. Levels given on the
command line replace the run's own.

```sh
uv run src/test.py test_run_id=<run id> test_iter=19600      # watch it in a window at 60 frames per second
uv run src/greedy_progress.py test_run_id=<run id>           # all checkpoints, without a window
uv run src/sampled_success.py test_run_id=<run id>           # success rate of each level
uv run src/clean_checkpoints.py <run id> [--apply]           # delete evaluated checkpoints that are no record
```

`greedy_progress.py` logs per checkpoint `greedy/progress`, the summed distance its episodes got, and
`greedy/levels_cleared`, tags the earliest best checkpoint as `best_greedy_iter` and plots the progress to `plots/`.
Calling it again only evaluates new checkpoints.

One greedy go tells little about how well the levels are learned: from one checkpoint to the next, it may clear all 32
levels or fail on the first. `sampled_success.py` measures a checkpoint's success rate on each level: 1024 episodes per
level, from small Mario, with actions sampled from the policy as in training, but without learning. It logs them as
`sampled/success_<level>`, with the checkpoint's iteration as step, and takes about 1¾ hours.

Checkpoints take about 72 MB; `clean_checkpoints.py` keeps those that
were not evaluated yet, each one that cleared more levels than all before it, and the tagged ones. Clean only runs that
will not be resumed: resuming restores the best checkpoint of the saved state, which may have been deleted since.

**World model previews.** Each DreamerV3 checkpoint stores `preview.gif`: replayed sequences of which the world model
sees the first quarter and predicts the rest from the actions taken, with the true and predicted frames and their
difference. `src/preview.py` renders the same for any checkpoint and level from episodes it plays itself, plus a
"dream" that continues with the agent's own actions, to `previews/`:

```sh
uv run src/preview.py test_run_id=<run id> levels=single levels.level=1-2
```

## Metrics

```sh
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

Metrics are logged once per iteration, with the iteration as MLflow's step. A **step** is one action of every worker,
held for 4 frames; an **iteration** is 128 steps of all 32 workers followed by the agent's update.

| Metric | Meaning |
|---|---|
| `levels/success_<level>`, `levels/distance_<level>` | share of the level's last 50 episodes that reached the flag, and the mean distance they got |
| `levels/cleared` | the expected number of levels cleared, the sum of the success rates |
| `levels/unlocked` | number of unlocked levels |
| `world_model/reconstruction`, `/reward`, `/dynamics` | DreamerV3: how well the world model predicts the frames, the rewards and its own next state |
| `behavior/entropy` | randomness of the policy, between 0 and ln 12 ≈ 2.5 |
| `behavior/cmpo_changes` | DreamerV3: share of replayed states where the CMPO target makes another action the most likely |
| `behavior/imagined_return`, `/return_scale` | DreamerV3: the return the actor expects in imagination, and the spread of returns that advantages are divided by |
| `exploration/plan2explore_reward` | Plan2Explore's reward per imagined step |
| `exploration/novelty_bonus`, `/visited_cells` | the novelty bonus per step, and the number of cells visited so far |
| `exploration/untried_actions` | share of the steps that took an untried action |
| `exploration/stuck_levels`, `/window_<level>` | levels stuck at a wall, and per level the window before the wall in pixels (0 while not stuck) |
| `powerups/start_<level>` | share of the level's episodes that start with a power-up |
| `progress/env_steps`, `/updates` | steps of all workers so far, and DreamerV3's updates so far |
| `greedy/progress`, `/levels_cleared` | per checkpoint, logged by `greedy_progress.py` |
| `sampled/success_<level>` | per checkpoint, logged by `sampled_success.py` |

## Project layout

| Path | Contents |
|---|---|
| `src/train.py`, `src/resume.py` | training runs, and resuming them |
| `src/environment.py`, `src/powerups.py` | the game, preprocessing and reward; the levels' start power-ups |
| `src/game_loop.py`, `src/exploration.py` | interaction between environments and agent, metrics; novelty bonus and untried actions |
| `src/level_schedulers.py` | the curriculum and the recent results of each level |
| `src/agents/` | the agent interface, `dreamer/` and `ppo/` |
| `src/evaluation.py`, `src/test.py`, `src/greedy_progress.py`, `src/sampled_success.py`, `src/clean_checkpoints.py`, `src/preview.py` | loading and playing trained agents |
| `src/config/` | dataclasses behind the Hydra configs |
| `test/` | tests; the PPO tests check the jaxtyping shape annotations at runtime |
