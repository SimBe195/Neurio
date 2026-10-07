"""
Hyperparameter search with Optuna. Each trial is a regular training run that is logged to mlflow, and unpromising
trials are stopped early based on the objective reported after each iteration.

Usage:
    uv run src/tune.py --study-name mechanics --search-space exploration --hours 10 --warmup-iters 250 -- \
        levels=tuning objective=mean_levels_cleared_score num_iters=700

Everything after `--` are Hydra overrides applied to every trial. The study is stored in `--storage`, so it can be
resumed by running the same command again, run from several processes at once, and inspected with
`uvx optuna-dashboard sqlite:///optuna.db`.
"""

import argparse
import logging
from pathlib import Path

import optuna
from hydra import compose, initialize_config_dir

from train import train

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def suggest_ppo(trial: optuna.Trial) -> list[str]:
    """
    How PPO learns from its experience, as Hydra overrides. Gamma, entropy and exploration are left at their defaults,
    which the exploration search space tuned.
    """
    return [
        f"agent.learning_rate.learning_rate={trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)}",
        f"agent.epochs_per_update={trial.suggest_int('epochs_per_update', 3, 10)}",
        f"agent.batch_size={trial.suggest_categorical('batch_size', [256, 512, 1024])}",
        f"agent.tau={trial.suggest_float('tau', 0.8, 0.99)}",
        f"agent.clip_param={trial.suggest_float('clip_param', 0.1, 0.3)}",
        f"agent.critic_loss_weight={trial.suggest_float('critic_loss_weight', 0.25, 1.0, log=True)}",
    ]


def suggest_exploration(trial: optuna.Trial) -> list[str]:
    """Self-imitation and curiosity jointly with the main PPO settings."""
    return [
        f"agent.sil_loss_weight={trial.suggest_float('sil_loss_weight', 0.1, 3.0, log=True)}",
        f"agent.sil_episodes_per_level={trial.suggest_int('sil_episodes_per_level', 4, 64, log=True)}",
        f"exploration.bonus_scale={trial.suggest_float('bonus_scale', 0.1, 2.0, log=True)}",
        f"agent.gamma={trial.suggest_float('gamma', 0.9, 0.995)}",
        f"agent.initial_entropy_loss_weight={trial.suggest_float('initial_entropy_loss_weight', 3e-3, 3e-2, log=True)}",
        f"agent.learning_rate.learning_rate={trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)}",
    ]


SEARCH_SPACES = {"ppo": suggest_ppo, "exploration": suggest_exploration}


def create_pruner(name: str, warmup_iters: int) -> optuna.pruners.BasePruner:
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=warmup_iters)
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=warmup_iters)
    if name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unknown pruner {name}")


def objective(trial: optuna.Trial, study_name: str, fixed_overrides: list[str], search_space: str = "ppo") -> float:
    overrides = fixed_overrides + SEARCH_SPACES[search_space](trial)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose("neurio_config", overrides=overrides)

    def on_iteration(iteration: int, value: float | None) -> bool:
        if value is None:
            return False
        trial.report(value, iteration)
        return trial.should_prune()

    result = train(
        cfg,
        overrides=overrides,
        on_iteration=on_iteration,
        tags={"optuna_study": study_name, "optuna_trial": str(trial.number)},
    )
    trial.set_user_attr("mlflow_run_id", result.run_id)
    if result.stopped_early:
        raise optuna.TrialPruned()
    return result.objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--search-space", choices=sorted(SEARCH_SPACES), default="ppo")
    parser.add_argument("--n-trials", type=int, default=None, help="Number of trials to run in this process")
    parser.add_argument("--hours", type=float, default=None, help="Stop starting new trials after this many hours")
    parser.add_argument("--storage", default="sqlite:///optuna.db")
    parser.add_argument("--pruner", choices=["median", "hyperband", "none"], default="median")
    parser.add_argument(
        "--warmup-iters", type=int, default=50, help="Iterations each trial runs before it can be pruned"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed of the sampler, different for each of several parallel processes"
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides for all trials")
    args = parser.parse_args()
    if args.n_trials is None and args.hours is None:
        parser.error("set --n-trials, --hours or both")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        # The constant liar treats running trials of parallel processes as bad, so that they do not all explore the
        # same region
        sampler=optuna.samplers.TPESampler(seed=args.seed, constant_liar=True),
        pruner=create_pruner(args.pruner, args.warmup_iters),
        load_if_exists=True,
    )
    # A NaN objective, i.e. no finished episode, marks the trial as failed without stopping the study
    # The time limit only stops new trials, a running trial finishes
    study.optimize(
        lambda trial: objective(trial, args.study_name, args.overrides, args.search_space),
        n_trials=args.n_trials,
        # A failing trial, e.g. out of memory, is marked as failed without stopping the study
        catch=(Exception,),
        timeout=args.hours * 3600 if args.hours is not None else None,
    )
    if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        log.info(f"Best trial {study.best_trial.number}: {study.best_value} with {study.best_params}")


if __name__ == "__main__":
    main()
