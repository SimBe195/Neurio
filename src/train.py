import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from agents import Agent, get_agent
from config import NeurioConfig, flatten
from environment import get_env_info, get_multiprocess_environment
from game_loop import GameLoop
from level_schedulers import get_level_scheduler
from novelty import NoveltyBonus

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

CONFIG_ARTIFACT = "config.yaml"

# Called after each iteration with the iteration (starting at 1) and the current objective. Returning True stops
# training early, e.g. to prune a hyperparameter search trial.
IterationCallback = Callable[[int, float | None], bool]


@dataclass
class TrainResult:
    objective: float
    run_id: str
    stopped_early: bool


class Checkpoints:
    """
    Saves checkpoints every `save_frequency` iterations, after the last iteration, and when the level scheduler's score
    improved by `min_improvement`, at most every `min_iters` iterations. The best one is noted in the run's tags, since
    later training can lose skills.
    """

    def __init__(self, agent: Agent, save_frequency: int, min_improvement: float = 0.1, min_iters: int = 25) -> None:
        self.agent = agent
        self.save_frequency = save_frequency
        self.min_improvement = min_improvement
        self.min_iters = min_iters
        self.best_score: float | None = None
        self.best_iter = -min_iters

    def after_iteration(self, train_iter: int, score: float | None, last: bool) -> None:
        improved = (
            score is not None
            and (self.best_score is None or score >= self.best_score + self.min_improvement)
            and train_iter - self.best_iter >= self.min_iters
        )
        if improved:
            self.best_score, self.best_iter = score, train_iter
            mlflow.set_tags({"best_checkpoint_iter": str(train_iter), "best_score": f"{score:.3f}"})
        if improved or last or train_iter % self.save_frequency == 0:
            self.agent.save(train_iter)


def get_experiment_name(config: NeurioConfig) -> str:
    return f"Neurio-lev-{config.levels.name}"


def load_run_config(run_id: str) -> DictConfig:
    """Load the config a run was trained with."""
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=CONFIG_ARTIFACT)
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    return cfg


def setup_mlflow(config: NeurioConfig) -> str:
    """Point mlflow to the configured tracking server and return the id of the experiment for the config's levels."""
    mlflow.set_tracking_uri(config.tracking_uri)
    name = get_experiment_name(config)
    if (experiment := mlflow.get_experiment_by_name(name)) is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name)


def current_objective(config: NeurioConfig, loop: GameLoop) -> float | None:
    """The configured objective, None while it is not available yet."""
    if config.objective == "levels_cleared_score":
        assert loop.level_scheduler is not None
        score = loop.level_scheduler.score()
        assert score is not None, "levels_cleared_score needs a level curriculum"
        return score
    return loop.recent_mean_reward


def train(
    cfg: DictConfig,
    overrides: list[str] | None = None,
    on_iteration: IterationCallback | None = None,
    tags: dict[str, str] | None = None,
) -> TrainResult:
    """
    Train an agent in a new mlflow run.

    :param cfg: Composed config, which is stored with the run so that the run can be reproduced and evaluated
    :param overrides: Overrides the config was composed with, stored as a tag for reference
    :param on_iteration: Optional callback after each iteration, see IterationCallback
    :param tags: Additional mlflow tags of the run
    :return: Final objective (see NeurioConfig.objective) and the run id
    """
    config: NeurioConfig = instantiate(cfg, _convert_="all")
    experiment_id = setup_mlflow(config)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        # The full config as artifact is the source of truth, the flattened params allow filtering in the UI
        container = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(container, dict)
        mlflow.log_dict({str(key): value for key, value in container.items()}, CONFIG_ARTIFACT)
        mlflow.log_params(flatten(asdict(config)))
        mlflow.set_tags({"overrides": " ".join(overrides or []), **config.tags, **(tags or {})})

        render_mode = "human" if config.render else None
        env = get_multiprocess_environment(config.num_workers, config.environment, config.levels, render_mode)
        try:
            agent = get_agent(config.agent, get_env_info(env))
            if (description := agent.describe()) is not None:
                mlflow.log_text(description, "model_summary.txt")
            novelty_bonus = NoveltyBonus(config.exploration) if config.exploration.bonus_scale > 0 else None
            loop = GameLoop(env, agent, get_level_scheduler(config.levels), config.objective_window, novelty_bonus)
            checkpoints = Checkpoints(agent, config.save_frequency)

            log.info(f"Train on levels {config.levels.name} for {config.num_iters} iters.")
            stopped_early = False
            for train_iter in tqdm(range(1, config.num_iters + 1)):
                loop.run_train_iter(config.steps_per_iter)
                stopped_early = on_iteration is not None and on_iteration(train_iter, current_objective(config, loop))
                last = stopped_early or train_iter == config.num_iters
                score = loop.level_scheduler.score() if loop.level_scheduler is not None else None
                checkpoints.after_iteration(train_iter, score, last)
                if stopped_early:
                    log.info(f"Stopped early after iter {train_iter}.")
                    mlflow.set_tag("stopped_early", "true")
                    break
        finally:
            env.close()

    objective = current_objective(config, loop)
    return TrainResult(math.nan if objective is None else objective, run.info.run_id, stopped_early)


@hydra.main(version_base="1.3", config_path="../configs", config_name="neurio_config")
def main(cfg: DictConfig) -> float:
    return train(cfg, overrides=list(HydraConfig.get().overrides.task)).objective


if __name__ == "__main__":
    main()
