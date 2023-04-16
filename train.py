import logging

import hydra
import mlflow
import optuna
from omegaconf import DictConfig
from tqdm import tqdm

from src.agents import Agent, get_agent_class
from src.environment import get_env_info, get_multiprocess_environment
from src.game_loop import GameLoop
from src.moving_average import MovingAverage

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    study_name = f"Neurio-lev-{config.level}"
    if (experiment := mlflow.get_experiment_by_name(study_name)) is not None:
        exp_id = experiment.experiment_id
    else:
        exp_id = mlflow.create_experiment(study_name)

    mlflow.set_tracking_uri(f"file://{hydra.utils.get_original_cwd()}/mlruns")
    with mlflow.start_run(experiment_id=exp_id):
        level = config.level

        # Create env
        train_env = get_multiprocess_environment(
            config.num_workers,
            config=config.environment,
            level=level,
            render_mode=None,
        )

        # Set up agent
        agent = get_agent_class(config.agent)(
            config.agent, get_env_info(train_env)
        )

        # Run game loop
        log.info(f"Train on level {level}")

        moving_reward_avg = MovingAverage(
            num_datapoints=10 * config.num_workers
        )
        loop = GameLoop(
            config, train_env, agent, reward_tracker=moving_reward_avg
        )

        log.info(f"Run {config.num_iters} iters.")
        mlflow.log_params(dict(config))
        for iter in tqdm(range(config.num_iters)):  # type: ignore
            loop.run_train_iter()
            if iter % config.save_frequency == 0:
                agent.save(iter)
            avg_reward = moving_reward_avg.get_value()
            mlflow.log_metric("avg_reward", avg_reward, step=iter)
        agent.save(config.num_iters)


if __name__ == "__main__":
    main()
