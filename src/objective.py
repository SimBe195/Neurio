import logging

import hydra
import mlflow
import optuna
from omegaconf import DictConfig
from tqdm import tqdm

from .agents import Agent, get_agent_class
from .environment import get_env_info, get_multiprocess_environment
from .game_loop import GameLoop
from .reward_trackers import MaxReward, MinReward, MovingAverageReward

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
            )

            # Set up agent
            agent = get_agent_class(self.config.agent)(
                self.config.agent, get_env_info(train_env), trial
            )

            # Run game loop
            log.info(f"Train on level {level}")

            max_reward = MaxReward()
            min_reward = MinReward()
            avg_reward = MovingAverageReward()
            loop = GameLoop(
                self.config, train_env, agent, reward_trackers=[max_reward, avg_reward, min_reward]
            )

            log.info(f"Run {self.config.num_iters} iters.")
            mlflow.log_params(trial.params)

            best_reward_avg = 0
            for iter in tqdm(range(1, self.config.num_iters + 1)):  # type: ignore
                loop.run_train_iter()
                self.maybe_save_model(iter, agent)
                mlflow.log_metric("min_reward", min_reward.get_value(), step=iter)
                mlflow.log_metric("max_reward", max_reward.get_value(), step=iter)
                mlflow.log_metric("avg_reward", avg_reward.get_value(), step=iter)
                best_reward_avg = max(best_reward_avg, avg_reward.get_value())
                trial.report(best_reward_avg, iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return best_reward_avg
