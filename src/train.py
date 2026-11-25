import logging
import tempfile
from dataclasses import asdict

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from agents import Agent, get_agent
from config import NeurioConfig, flatten
from environment import get_env_info, get_multiprocess_environment
from exploration import NoveltyBonus, UntriedActions
from game_loop import GameLoop
from level_schedulers import get_level_scheduler
from powerups import StartPowerups
from resume import (
    BEST_CHECKPOINT_ITER_TAG,
    BEST_SCORE_TAG,
    CONFIG_HASH_TAG,
    config_hash,
    decide_resume,
    discard_after,
    find_previous_run,
    load_training_state,
    save_training_state,
)

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

CONFIG_ARTIFACT = "config.yaml"


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

    def after_iteration(self, train_iter: int, score: float, last: bool) -> None:
        improved = (
            self.best_score is None or score >= self.best_score + self.min_improvement
        ) and train_iter - self.best_iter >= self.min_iters
        if improved:
            self.best_score, self.best_iter = score, train_iter
            self.tag_best()
        if improved or last or train_iter % self.save_frequency == 0:
            self.agent.save(train_iter)

    def tag_best(self) -> None:
        if self.best_score is not None:
            mlflow.set_tags({BEST_CHECKPOINT_ITER_TAG: str(self.best_iter), BEST_SCORE_TAG: f"{self.best_score:.3f}"})

    def state_dict(self) -> dict:
        return {"best_score": self.best_score, "best_iter": self.best_iter}

    def load_state_dict(self, state: dict) -> None:
        self.best_score, self.best_iter = state["best_score"], state["best_iter"]


def get_experiment_name(config: NeurioConfig) -> str:
    return f"Neurio-lev-{config.levels.name}"


def load_run_config(run_id: str) -> DictConfig:
    """Load the config a run was trained with."""
    with tempfile.TemporaryDirectory() as directory:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=CONFIG_ARTIFACT, dst_path=directory)
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


def train(cfg: DictConfig, overrides: list[str] | None = None, resume_run_id: str | None = None) -> str:
    """
    Train an agent in a new mlflow run, or continue an interrupted one.

    :param cfg: Composed config, which is stored with the run so that the run can be reproduced and evaluated
    :param overrides: Overrides the config was composed with, stored as a tag for reference
    :param resume_run_id: Run to continue from its last checkpoint, which must have been trained with the same config
    :return: The id of the run
    """
    config: NeurioConfig = instantiate(cfg, _convert_="all")
    experiment_id = setup_mlflow(config)

    with mlflow.start_run(run_id=resume_run_id, experiment_id=None if resume_run_id else experiment_id) as run:
        if resume_run_id is None:
            # The full config as artifact is the source of truth, the flattened params allow filtering in the UI
            container = OmegaConf.to_container(cfg, resolve=True)
            assert isinstance(container, dict)
            mlflow.log_dict({str(key): value for key, value in container.items()}, CONFIG_ARTIFACT)
            mlflow.log_params(flatten(asdict(config)))
            mlflow.set_tags({"overrides": " ".join(overrides or []), CONFIG_HASH_TAG: config_hash(cfg), **config.tags})

        env = get_multiprocess_environment(
            config.num_workers, config.environment, config.levels, config.agent.stack_frames
        )
        try:
            env_info = get_env_info(env)
            agent = get_agent(config.agent, env_info)
            if (description := agent.describe()) is not None:
                mlflow.log_text(description, "model_summary.txt")
            exploration = config.exploration
            level_scheduler = get_level_scheduler(config.levels)
            loop = GameLoop(
                env,
                agent,
                level_scheduler,
                novelty_bonus=NoveltyBonus(exploration),
                start_powerups=StartPowerups(),
                untried_actions=UntriedActions(
                    exploration.cell_size, env_info.num_actions, config.levels.success_window
                ),
            )
            checkpoints = Checkpoints(agent, config.save_frequency)
            first_iter = 1
            if resume_run_id is not None:
                last_iter, state = load_training_state(resume_run_id)
                loop.load_state_dict(state["loop"])
                checkpoints.load_state_dict(state["checkpoints"])
                # Training may have gone on after the state was saved, which the resumed run repeats
                discard_after(resume_run_id, last_iter)
                checkpoints.tag_best()
                first_iter = last_iter + 1
                mlflow.set_tag("resumed_after_iter", str(last_iter))

            log.info(f"Train on levels {config.levels.name} from iter {first_iter} to {config.num_iters}.")
            iters = range(first_iter, config.num_iters + 1)
            for train_iter in tqdm(iters, initial=first_iter - 1, total=config.num_iters):
                loop.run_train_iter(config.steps_per_iter)
                last = train_iter == config.num_iters
                checkpoints.after_iteration(train_iter, level_scheduler.score(), last)
                if last or train_iter % config.resume_frequency == 0:
                    save_training_state(
                        train_iter, {"loop": loop.state_dict(), "checkpoints": checkpoints.state_dict()}
                    )
        finally:
            env.close()

    return run.info.run_id


@hydra.main(version_base="1.3", config_path="../configs", config_name="neurio_config")
def main(cfg: DictConfig) -> None:
    config: NeurioConfig = instantiate(cfg, _convert_="all")
    previous = find_previous_run(setup_mlflow(config), config_hash(cfg), config.num_iters)
    resume_run_id = decide_resume(previous, config.resume)
    train(cfg, overrides=list(HydraConfig.get().overrides.task), resume_run_id=resume_run_id)


if __name__ == "__main__":
    main()
