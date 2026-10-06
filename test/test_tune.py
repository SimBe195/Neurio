import optuna
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import tune
from level_schedulers import get_level_scheduler
from train import TrainResult


def test_search_space_composes_valid_configs():
    params = {
        "learning_rate": 3e-4,
        "gamma": 0.97,
        "tau": 0.9,
        "critic_loss_weight": 0.3,
        "initial_entropy_loss_weight": 0.01,
        "batch_size": 1024,
        "epochs_per_update": 4,
    }
    overrides = tune.suggest_ppo(optuna.trial.FixedTrial(params))
    with initialize_config_dir(config_dir=tune.CONFIG_DIR, version_base="1.3"):
        config = instantiate(compose("neurio_config", overrides=overrides), _convert_="all")

    agent = config.agent
    assert agent.learning_rate.learning_rate == pytest.approx(3e-4)
    assert (agent.gamma, agent.tau, agent.critic_loss_weight) == pytest.approx((0.97, 0.9, 0.3))
    assert agent.initial_entropy_loss_weight == pytest.approx(0.01)
    assert (agent.batch_size, agent.epochs_per_update) == (1024, 4)


def test_exploration_search_space_composes_valid_configs():
    params = {
        "sil_loss_weight": 1.0,
        "sil_episodes_per_level": 16,
        "bonus_scale": 0.5,
        "gamma": 0.99,
        "initial_entropy_loss_weight": 0.01,
        "learning_rate": 7e-4,
    }
    overrides = [*tune.suggest_exploration(optuna.trial.FixedTrial(params)), "objective=levels_cleared_score"]
    with initialize_config_dir(config_dir=tune.CONFIG_DIR, version_base="1.3"):
        config = instantiate(compose("neurio_config", overrides=overrides), _convert_="all")

    assert (config.agent.sil_loss_weight, config.agent.sil_episodes_per_level) == (1.0, 16)
    assert config.exploration.bonus_scale == pytest.approx(0.5)
    assert config.agent.learning_rate.learning_rate == pytest.approx(7e-4)
    assert config.levels.name == "curriculum-32"
    assert config.objective == "levels_cleared_score"


class AlwaysPrune(optuna.pruners.BasePruner):
    def prune(self, study, trial) -> bool:
        return True


def _run_trial(monkeypatch, pruner, reported):
    calls = {}

    def fake_train(cfg, overrides, on_iteration, tags):
        calls.update(overrides=overrides, tags=tags, cfg=cfg)
        stopped = False
        for iteration, value in enumerate(reported, start=1):
            if on_iteration(iteration, value):
                stopped = True
                break
        return TrainResult(objective=7.0, run_id="run-123", stopped_early=stopped)

    monkeypatch.setattr(tune, "train", fake_train)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial: tune.objective(trial, "my-study", ["levels=single"], "exploration"), n_trials=1)
    return study.trials[0], calls


def test_objective_completes_trial(monkeypatch):
    trial, calls = _run_trial(monkeypatch, optuna.pruners.NopPruner(), [None, 1.0, 2.0])

    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert trial.value == 7.0
    assert trial.user_attrs["mlflow_run_id"] == "run-123"
    # Iterations without any finished episode are not reported
    assert trial.intermediate_values == {2: 1.0, 3: 2.0}
    assert calls["overrides"][0] == "levels=single" and len(calls["overrides"]) == 7
    assert calls["tags"] == {"optuna_study": "my-study", "optuna_trial": "0"}
    assert calls["cfg"].levels.level == "1-1"


def test_objective_prunes_trial(monkeypatch):
    trial, _ = _run_trial(monkeypatch, AlwaysPrune(), [1.0, 2.0])
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert trial.intermediate_values == {1: 1.0}
    assert trial.user_attrs["mlflow_run_id"] == "run-123"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("median", optuna.pruners.MedianPruner),
        ("hyperband", optuna.pruners.HyperbandPruner),
        ("none", optuna.pruners.NopPruner),
    ],
)
def test_create_pruner(name, expected):
    assert isinstance(tune.create_pruner(name, warmup_iters=10), expected)


def test_tuning_levels_are_a_curriculum():
    with initialize_config_dir(config_dir=tune.CONFIG_DIR, version_base="1.3"):
        config = instantiate(compose("neurio_config", overrides=["levels=tuning"]), _convert_="all")
    assert config.levels.levels == ["1-1", "1-3", "2-2", "4-1"]
    assert get_level_scheduler(config.levels).distribution()[0] == ["1-1"]
