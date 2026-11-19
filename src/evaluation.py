"""Loading a trained agent for evaluation, as test.py and preview.py do."""

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields

import mlflow
import pandas
import torch
from gymnasium.vector import AsyncVectorEnv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from agents import CHECKPOINT_PREFIX, Agent, get_agent
from config import NeurioConfig
from environment import Powerup, RenderMode, get_env_info, get_multiprocess_environment
from game_loop import EpisodeStats, GameLoop
from train import CONFIG_ARTIFACT, get_experiment_name, load_run_config

log = logging.getLogger(__name__)

# Seed of each evaluation episode, so that an agent always plays a level the same way: with the same random start
# frames and the same sampled model states
EVALUATION_SEED = 0


def find_best_run_id(config: NeurioConfig) -> str:
    """The run of the configured agent with the best final objective in the experiment of the configured level mode."""
    runs = mlflow.search_runs(experiment_names=[get_experiment_name(config)], output_format="pandas")
    assert isinstance(runs, pandas.DataFrame)
    # Runs of other agents lack the configured agent's settings
    for field in fields(config.agent):
        param = f"params.agent.{field.name}"
        # Nested settings are logged by their own names
        logged = [column for column in runs if column == param or column.startswith(f"{param}.")]
        runs = runs[runs[logged].notna().any(axis=1)] if logged else runs.iloc[:0]
    metric = f"metrics.levels/{config.objective}"
    assert metric in runs and runs[metric].notna().any(), f"No runs of this agent with {config.objective} found."
    best = runs.loc[runs[metric].idxmax()]
    log.info(f"Best run is {best['run_id']} with {config.objective} {best[metric]:.3f}.")
    return best["run_id"]


def checkpoint_iters(run_id: str) -> list[int]:
    """The iterations of the run's checkpoints, in order."""
    pattern = re.compile(re.escape(CHECKPOINT_PREFIX) + r"(\d+)")
    return sorted(
        int(match.group(1))
        for artifact in mlflow.MlflowClient().list_artifacts(run_id)
        if (match := pattern.fullmatch(artifact.path))
    )


def latest_checkpoint_iter(run_id: str) -> int:
    iters = checkpoint_iters(run_id)
    assert iters, f"Run {run_id} has no checkpoints."
    return iters[-1]


def play_levels(loop: GameLoop, levels: list[str]) -> list[EpisodeStats]:
    """
    Let the agent of a greedy game loop with one worker play the levels in order, until the first one without the
    flag, like the game: Mario starts small and keeps the power-up with which he reached the flag for the next level.
    Episodes are seeded, see EVALUATION_SEED.

    :return: The episodes played, the last one without the flag unless all levels were cleared
    """
    episodes: list[EpisodeStats] = []
    powerup = Powerup.SMALL
    for level in levels:
        loop.env.call("set_level_distribution", [level], [1.0])
        loop.env.call("set_start_powerups", {level: [float(p == powerup) for p in Powerup]})
        torch.manual_seed(EVALUATION_SEED)
        loop.restart(seed=EVALUATION_SEED)
        episodes.append(loop.run_test_episode())
        if not episodes[-1].flag_get:
            break
        powerup = episodes[-1].powerup
    return episodes


def evaluation_config(run_id: str, cli_cfg: DictConfig, levels_overridden: bool) -> NeurioConfig:
    """The run's own config for a single worker, on the levels from the command line if given, else on its own."""
    try:
        run_cfg = load_run_config(run_id)
    except Exception as e:
        raise RuntimeError(f"Could not load {CONFIG_ARTIFACT} of run {run_id}, runs of older versions lack it.") from e
    # Settings added since, which only concern training, e.g. resume_frequency
    for key in cli_cfg:
        if key not in run_cfg:
            run_cfg[key] = cli_cfg[key]
    run_cfg.num_workers = 1
    # Without sticky actions, so that the agent's most likely actions always play the same way
    run_cfg.environment.sticky_action_probability = 0.0
    if levels_overridden:
        run_cfg.levels = cli_cfg.levels
    return instantiate(run_cfg, _convert_="all")


@dataclass
class Evaluation:
    config: NeurioConfig
    run_id: str
    iteration: int
    env: AsyncVectorEnv
    agent: Agent


@contextmanager
def load_evaluation(cfg: DictConfig, render_mode: RenderMode | None = None) -> Iterator[Evaluation]:
    """
    The agent of the run `test_run_id`, by default the best of the experiment of the configured level mode, at the
    checkpoint `test_iter`, by default the latest one, with an environment of one worker. It plays the evaluation
    levels (see LevelConfig.evaluation_levels) of those from the command line, by default of the run's own, each
    equally likely.
    """
    cli_config: NeurioConfig = instantiate(cfg, _convert_="all")
    mlflow.set_tracking_uri(cli_config.tracking_uri)
    run_id = cli_config.test_run_id or find_best_run_id(cli_config)
    levels_overridden = any(o.lstrip("+~").startswith("levels") for o in HydraConfig.get().overrides.task)
    config = evaluation_config(run_id, cfg, levels_overridden)
    env = get_multiprocess_environment(1, config.environment, config.levels, config.agent.stack_frames, render_mode)
    try:
        levels = config.levels.evaluation_levels()
        env.call("set_level_distribution", levels, [1 / len(levels)] * len(levels))
        agent = get_agent(config.agent, get_env_info(env))
        iteration = cli_config.test_iter if cli_config.test_iter is not None else latest_checkpoint_iter(run_id)
        log.info(f"Run {run_id} at iteration {iteration} on levels {config.levels.name}.")
        agent.load(run_id, iteration)
        yield Evaluation(config, run_id, iteration, env, agent)
    finally:
        env.close()
