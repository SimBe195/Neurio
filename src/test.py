import logging
import re

import hydra
import mlflow
import pandas
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from agents import get_agent
from config import NeurioConfig
from environment import get_env_info, get_multiprocess_environment
from game_loop import GameLoop
from train import CONFIG_ARTIFACT, get_experiment_name, load_run_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def find_best_run_id(config: NeurioConfig) -> str:
    """The run with the best final objective in the experiment of the configured level mode."""
    runs = mlflow.search_runs(experiment_names=[get_experiment_name(config)], output_format="pandas")
    assert isinstance(runs, pandas.DataFrame)
    metric = f"metrics.{config.objective}"
    assert metric in runs and runs[metric].notna().any(), f"No runs with {config.objective} found."
    best = runs.loc[runs[metric].idxmax()]
    log.info(f"Best run is {best['run_id']} with {config.objective} {best[metric]:.3f}.")
    return best["run_id"]


def latest_checkpoint_iter(run_id: str) -> int:
    pattern = re.compile(r"actor_critic-(\d+)")
    iters = [
        int(match.group(1))
        for artifact in mlflow.MlflowClient().list_artifacts(run_id)
        if (match := pattern.fullmatch(artifact.path))
    ]
    assert iters, f"Run {run_id} has no checkpoints."
    return max(iters)


def evaluation_config(run_id: str, cli_cfg: DictConfig, levels_overridden: bool) -> NeurioConfig:
    """
    The run's own config for a single, greedily acting worker, on the levels from the command line if given, else on
    the full game.
    """
    try:
        run_cfg = load_run_config(run_id)
    except Exception as e:
        raise RuntimeError(f"Could not load {CONFIG_ARTIFACT} of run {run_id}, runs of older versions lack it.") from e
    run_cfg.num_workers = 1
    if levels_overridden:
        run_cfg.levels = cli_cfg.levels
    else:
        run_cfg.levels = OmegaConf.create({"_target_": "config.levels.ProgressionLevelsConfig"})
    if "sampling_strategy" in run_cfg.agent:
        run_cfg.agent.sampling_strategy = OmegaConf.create(
            {"_target_": "config.sampling_strategy.GreedySamplingStrategy"}
        )
    return instantiate(run_cfg, _convert_="all")


@hydra.main(version_base="1.3", config_path="../configs", config_name="neurio_config")
def main(cfg: DictConfig) -> None:
    """
    Watch a trained agent play one episode. By default, it plays the full game from 1-1 and continues after each flag
    until Mario loses a life; levels given on the command line replace that, e.g. `levels=single levels.level=1-3`.
    The agent always takes its most likely action, so the episode is deterministic.

    Plays the best run of the experiment of the configured level mode, or `test_run_id`, at the checkpoint `test_iter`,
    by default the latest one.
    """
    cli_config: NeurioConfig = instantiate(cfg, _convert_="all")
    mlflow.set_tracking_uri(cli_config.tracking_uri)
    run_id = cli_config.test_run_id or find_best_run_id(cli_config)
    levels_overridden = any(o.lstrip("+~").startswith("levels") for o in HydraConfig.get().overrides.task)
    config = evaluation_config(run_id, cfg, levels_overridden)

    env = get_multiprocess_environment(1, config.environment, config.levels, render_mode="human")
    try:
        if (distribution := config.levels.evaluation_distribution()) is not None:
            env.call("set_level_distribution", *distribution)
        agent = get_agent(config.agent, get_env_info(env))
        test_iter = cli_config.test_iter if cli_config.test_iter is not None else latest_checkpoint_iter(run_id)
        log.info(f"Eval run {run_id} at iter {test_iter} on levels {config.levels.name}.")
        agent.load(run_id, test_iter)

        episode = GameLoop(env, agent).run_test_episode()
        result = "with the flag" if episode.flag_get else "without the flag"
        log.info(f"Ended on {episode.level} at x {episode.x_pos} {result}, reward {episode.reward:.1f}.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
