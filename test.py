import logging

import hydra
import mlflow

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

import optuna
from omegaconf import DictConfig

from src.agents import get_agent_class
from src.environment import get_env_info, get_singleprocess_environment
from src.game_loop import GameLoop


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    study_name = f"Neurio-lev-{config.level}"

    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///optuna_studies/{study_name}.db",
    )

    trial = study.get_trials()[config.test_trial]

    level = config.level

    # Create env
    env = get_singleprocess_environment(
        config=config.environment,
        level=level,
        render_mode="human",
    )

    # Set up agent
    agent = get_agent_class(config.agent)(config.agent, get_env_info(env), trial)

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

        log.info(f"Loading checkpoint at iter {config.test_iter}.")
        agent.load(config.test_iter)

        GameLoop(config, env, agent).run_test_loop()

        log.info("Done evaluating. Exiting now.")


if __name__ == "__main__":
    main()
