import logging
from dataclasses import asdict

import hydra
import mlflow
from hydra.utils import instantiate
from tqdm import tqdm

from src.agents import get_agent
from src.config import NeurioConfig, flatten
from src.environment import get_env_info, get_multiprocess_environment
from src.game_loop import GameLoop
from src.reward_trackers import (
    BestReward,
    MovingAvgReward,
    MovingMaxReward,
    MovingMinReward,
    SumReward,
)

logging.basicConfig(level=logging.INFO)

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
        mlflow.log_params(flatten(asdict(config)))
        level = config.level

        # Create env
        train_env = get_multiprocess_environment(
            config.num_workers,
            config=config.environment,
            level=level,
            render_mode="human" if config.render else None,
        )

        # Set up agent
        agent = get_agent(config.agent, get_env_info(train_env))

        # Run game loop
        log.info(f"Train on level {level}")

        max_reward = MovingMaxReward(history_size=500)
        min_reward = MovingMinReward(history_size=500)
        avg_rewards = {}
        for hist_size in [100, 200, 500, 1000, 2000, 5000, 10000, None]:
            if hist_size is not None:
                name = f"avg_reward_{hist_size:05d}"
            else:
                name = "avg_reward"
            avg_rewards[name] = MovingAvgReward(history_size=hist_size)
        sum_avg_rewards = SumReward(list(avg_rewards.values()))
        best_sum_avg_rewards = BestReward(sum_avg_rewards)
        loop = GameLoop(
            train_env,
            agent,
            reward_trackers={
                "min_reward": min_reward,
                "max_reward": max_reward,
                **avg_rewards,
                "sum_avg_rewards": sum_avg_rewards,
                "best_sum_avg_rewards": best_sum_avg_rewards,
            },
        )

        log.info(f"Run {config.num_iters} iters.")
        for iter in tqdm(range(config.num_iters)):
            loop.run_train_iter(config.steps_per_iter)
            if iter % config.save_frequency == 0:
                agent.save(iter)
        agent.save(config.num_iters)

        return best_sum_avg_rewards.get_value()


if __name__ == "__main__":
    main()  # type: ignore
