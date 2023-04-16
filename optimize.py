import logging

import hydra
import mlflow
import optuna

logging.basicConfig(level=logging.DEBUG)

from omegaconf import DictConfig

from src.objective import Objective


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    study_name = f"Neurio-lev-{config.level}"
    if (experiment := mlflow.get_experiment_by_name(study_name)) is not None:
        exp_id = experiment.experiment_id
    else:
        exp_id = mlflow.create_experiment(study_name)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        load_if_exists=True,
        storage=f"sqlite:///optuna_studies/{study_name}.db",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    )
    study.optimize(Objective(config, exp_id), n_trials=10)

    logging.info(f"Best params: {study.best_params}")
    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best trial: {study.best_trial}")


if __name__ == "__main__":
    main()
