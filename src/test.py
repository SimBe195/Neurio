import logging

import hydra
import mlflow
import pandas
from hydra.utils import instantiate

from agents import get_agent
from config import NeurioConfig, unflatten
from config import update_config_with_dict
from environment import get_env_info, get_multiprocess_environment
from game_loop import GameLoop

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="neurio_config")
def main(config: NeurioConfig) -> None:
    config = instantiate(config)
    if config.test_run_id is not None:
        run = mlflow.get_run(config.test_run_id)
        log.info(f"Loading run with run_id {run.info.run_id}")
    else:
        runs_df = mlflow.search_runs(
            experiment_names=[f"Neurio-lev-{config.level}"],
            output_format="pandas",
        )
        assert isinstance(runs_df, pandas.DataFrame)
        log.info("Finding best run.")
        rewards = runs_df["metrics.best_sum_avg_rewards"]
        run = mlflow.get_run(runs_df.loc[rewards.idxmax()]["run_id"])
        log.info(f"Loading best run with run_id {run.info.run_id} and metric {rewards.max()}.")

    with mlflow.start_run(run_id=run.info.run_id):
        log.info("Update config with run parameters.")
        run_params = unflatten(run.data.params)
        run_params.pop("test_iter")
        update_config_with_dict(config, run_params, log)

        log.info("Set num_workers to 1")
        config.num_workers = 1

        level = config.level

        # Create env
        env = get_multiprocess_environment(
            num_environments=config.num_workers,
            config=config.environment,
            level=level,
            render_mode="human",
        )

        # Set up agent
        agent = get_agent(config.agent, get_env_info(env))

        # Load checkpoint
        log.info(f"Eval trained model on level {level}.")
        if config.test_iter is not None:
            log.info(f"Loading checkpoint at iter {config.test_iter}.")
            agent.load(config.test_iter)
        else:
            log.info(f"Loading checkpoint at iter {config.num_iters}.")
            agent.load(config.num_iters)

        GameLoop(env, agent).run_test_loop(framerate=60 // config.environment.num_repeat_frames)

        log.info("Done evaluating. Exiting now.")


if __name__ == "__main__":
    main()
