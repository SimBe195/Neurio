import logging

import hydra
import mlflow
import optuna
from omegaconf import DictConfig
from tqdm import tqdm

from .agents import Agent, get_agent_class
from .environment import get_env_info, get_multiprocess_environment
from .game_loop import GameLoop
from .moving_average import MovingAverage

log = logging.getLogger(__name__)


class Objective:
    def __init__(self, config: DictConfig, exp_id: str):
        self.config = config
        self.exp_id = exp_id

    def maybe_save_model(self, save_iter: int, agent: Agent) -> None:
        if save_iter % self.config.save_frequency != 0:
            return

        agent.save(save_iter)

    def __call__(self, trial: optuna.Trial) -> float:
        mlflow.set_tracking_uri(f"file://{hydra.utils.get_original_cwd()}/mlruns")
        with mlflow.start_run(
            experiment_id=self.exp_id, run_name=f"trial-{trial.number:04d}"
        ):
            level = self.config.level

            # Create env
            train_env = get_multiprocess_environment(
                self.config.num_workers,
                config=self.config.environment,
                level=level,
                render_mode=None,
                trial=trial,
            )

            # Set up agent
            agent = get_agent_class(self.config.agent)(
                self.config.agent, get_env_info(train_env), trial
            )

            # Run game loop
            log.info(f"Train on level {level}")

            moving_reward_avg = MovingAverage(
                num_datapoints=10 * self.config.num_workers
            )
            loop = GameLoop(
                self.config, train_env, agent, reward_tracker=moving_reward_avg
            )

            log.info(f"Run {self.config.num_iters} iters.")
            mlflow.log_params(trial.params)
            for iter in tqdm(range(self.config.num_iters)):  # type: ignore
                loop.run_train_iter()
                self.maybe_save_model(iter, agent)
                avg_reward = moving_reward_avg.get_value()
                mlflow.log_metric("avg_reward", avg_reward, step=iter)
                trial.report(avg_reward, iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return moving_reward_avg.get_value()
