import logging
from dataclasses import asdict

import hydra
import mlflow
from hydra.utils import instantiate
from tqdm import tqdm

from src.agents import get_agent
from src.config import NeurioConfig
from src.environment import get_env_info, get_multiprocess_environment
from src.game_loop import GameLoop
from src.reward_trackers import (
    BestMovingAvgReward,
    MovingAvgReward,
    MovingMaxReward,
    MovingMinReward,
)

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="neurio_config")
def main(config: NeurioConfig) -> float:
    config = instantiate(config)
    study_name = f"Neurio-lev-{config.level}"
    if (experiment := mlflow.get_experiment_by_name(study_name)) is not None:
        exp_id = experiment.experiment_id
    else:
        exp_id = mlflow.create_experiment(study_name)

    mlflow.set_tracking_uri(f"file://{hydra.utils.get_original_cwd()}/mlruns")
    with mlflow.start_run(experiment_id=exp_id):
        mlflow.log_params(asdict(config))
        level = config.level

        # Create env
        train_env = get_multiprocess_environment(
            config.num_workers,
            config=config.environment,
            level=level,
            render_mode=None,
        )

        # Set up agent
        agent = get_agent(config.agent, get_env_info(train_env))

        # Run game loop
        log.info(f"Train on level {level}")

        max_reward = MovingMaxReward()
        min_reward = MovingMinReward()
        avg_reward = MovingAvgReward()
        best_avg_reward = BestMovingAvgReward()
        loop = GameLoop(
            train_env,
            agent,
            reward_trackers=[min_reward, max_reward, avg_reward, best_avg_reward],
        )

        log.info(f"Run {config.num_iters} iters.")
        for iter in tqdm(range(config.num_iters)):
            loop.run_train_iter(config.steps_per_iter)
            if iter % config.save_frequency == 0:
                agent.save(iter)
            mlflow.log_metric("min_reward", min_reward.get_value(), step=iter)
            mlflow.log_metric("max_reward", max_reward.get_value(), step=iter)
            mlflow.log_metric("avg_reward", avg_reward.get_value(), step=iter)
            mlflow.log_metric("best_avg_reward", best_avg_reward.get_value(), step=iter)
        agent.save(config.num_iters)

        return best_avg_reward.get_value()


if __name__ == "__main__":
    main()  # type: ignore
