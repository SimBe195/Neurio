import argparse
import logging

import hydra
import mlflow

from config.main_config import NeurioConfig

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

import optuna

from src.agents import get_agent
from src.environment import get_env_info, get_singleprocess_environment
from src.game_loop import GameLoop


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: NeurioConfig) -> None:
    parser = argparse.ArgumentParser(description="Run test for model.")
    parser.add_argument("--trial", type=int, help="Number of test trial to load.")
    parser.add_argument("--iter", type=int, help="Number of test iter to load.")

    args = parser.parse_args()

    study_name = f"Neurio-lev-{config.level}"

    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///optuna_studies/{study_name}.db",
    )

    trial = study.get_trials()[args.trial]

    level = config.level

    # Create env
    env = get_singleprocess_environment(
        config=config.environment, level=level, render_mode="human"
    )

    # Set up agent
    agent = get_agent(config.agent, get_env_info(env))

    runs = mlflow.search_runs(
        experiment_names=[study_name],
        filter_string=f"run_name='trial-{trial.number:04d}'",
        output_format="list",
    )
    assert isinstance(runs, list)
    if not runs:
        return
    run_id = runs[0].info.run_id
    logging.info(f"Found run with run_id {run_id}")

    with mlflow.start_run(run_id=run_id):
        # Load checkpoint
        log.info(f"Eval trained model on level {level}.")

        log.info(f"Loading checkpoint at iter {args.iter}.")
        agent.load(args.iter)

        GameLoop(env, agent).run_test_loop()

        log.info("Done evaluating. Exiting now.")


if __name__ == "__main__":
    main()
