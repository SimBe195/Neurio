import importlib.util
from pathlib import Path

import mlflow
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import environment
import train as train_module
from agents import get_agent
from config.levels import ProgressionLevelsConfig
from config.sampling_strategy import GreedySamplingStrategy
from environment import get_env_info, get_environment
from fakes import FakeMarioEnv
from test_game_loop import make_vector_env
from train import CONFIG_ARTIFACT, Checkpoints, load_run_config, train

ROOT = Path(__file__).resolve().parents[1]

# src/test.py would clash with the test directory and Python's own test package when imported by name
_spec = importlib.util.spec_from_file_location("neurio_test_script", ROOT / "src" / "test.py")
test_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(test_script)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    # mlflow stores artifacts relative to the working directory
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        environment, "BaseEnvironment", lambda config, level, **_: FakeMarioEnv([10.0] * 3, level=level)
    )
    monkeypatch.setattr(
        train_module,
        "get_multiprocess_environment",
        lambda num, config, levels, render_mode: make_vector_env([get_environment(config, levels) for _ in range(num)]),
    )
    overrides = [
        "levels=single",
        "num_workers=2",
        "num_iters=3",
        "steps_per_iter=4",
        "save_frequency=2",
        "agent.batch_size=4",
        f"tracking_uri=sqlite:///{tmp_path}/mlflow.db",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose("neurio_config", overrides=overrides), overrides


def _artifacts(run_id: str) -> list[str]:
    return sorted(a.path for a in mlflow.MlflowClient().list_artifacts(run_id))


def test_train_logs_run(cfg):
    cfg, overrides = cfg
    cfg.tags = {"from_config": "yes"}
    result = train(cfg, overrides=overrides, tags={"purpose": "test"})

    # Episodes of 3 steps with rewards of 10 per step, scaled by 0.1
    assert result.objective == pytest.approx(3.0)
    assert not result.stopped_early

    run = mlflow.get_run(result.run_id)
    assert run.data.tags["overrides"] == " ".join(overrides)
    assert run.data.tags["purpose"] == "test"
    assert run.data.tags["from_config"] == "yes"
    assert run.data.params["agent.gamma"] == "0.99"
    assert {CONFIG_ARTIFACT, "actor_critic-0002", "actor_critic-0003"} <= set(_artifacts(result.run_id))

    history = mlflow.MlflowClient().get_metric_history(result.run_id, "recent_mean_reward")
    # Stepped by the environment steps of both workers
    assert [m.step for m in history] == [8, 16, 24]
    assert history[-1].value == pytest.approx(result.objective)


def test_stored_config_reproduces_the_run(cfg):
    cfg, overrides = cfg
    result = train(cfg, overrides=overrides)

    stored = load_run_config(result.run_id)
    assert OmegaConf.to_container(stored) == OmegaConf.to_container(cfg, resolve=True)
    assert instantiate(stored, _convert_="all") == instantiate(cfg, _convert_="all")


def test_train_stops_early(cfg):
    cfg, overrides = cfg
    seen = []

    def on_iteration(iteration, objective):
        seen.append((iteration, objective))
        return iteration == 2

    result = train(cfg, overrides=overrides, on_iteration=on_iteration)

    assert result.stopped_early
    assert [iteration for iteration, _ in seen] == [1, 2]
    assert all(objective == pytest.approx(3.0) for _, objective in seen)
    assert mlflow.get_run(result.run_id).data.tags["stopped_early"] == "true"
    assert test_script.latest_checkpoint_iter(result.run_id) == 2


def test_find_best_run(cfg):
    cfg, overrides = cfg
    first = train(cfg, overrides=overrides)
    cfg.environment.reward_scale = 1.0
    second = train(cfg, overrides=overrides)
    assert second.objective > first.objective
    assert test_script.find_best_run_id(instantiate(cfg, _convert_="all")) == second.run_id


def test_curriculum_keeps_best_checkpoint(cfg, monkeypatch):
    cfg, overrides = cfg
    monkeypatch.setattr(
        environment, "BaseEnvironment", lambda config, level, **_: FakeMarioEnv([10.0] * 3, level=level, flag_get=True)
    )
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg.levels = compose(
            "neurio_config",
            overrides=[
                "levels=curriculum",
                "levels.levels=[1-1,1-2]",
                "levels.success_window=2",
                "levels.min_level_episodes=1",
            ],
        ).levels
    result = train(cfg, overrides=overrides)

    run = mlflow.get_run(result.run_id)
    # The first iteration already sets the best score, later ones are within the minimum distance between saves
    assert run.data.tags["best_checkpoint_iter"] == "1"
    assert float(run.data.tags["best_score"]) > 0
    assert "actor_critic-0001" in _artifacts(result.run_id)
    # 1-1 is cleared, 1-2 depends on whether it was sampled after its unlock
    assert run.data.metrics["levels_cleared_score"] >= 1.0


def test_levels_cleared_objective(cfg, monkeypatch):
    cfg, overrides = cfg
    monkeypatch.setattr(
        environment, "BaseEnvironment", lambda config, level, **_: FakeMarioEnv([10.0] * 3, level=level, flag_get=True)
    )
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg.levels = compose(
            "neurio_config",
            overrides=[
                "levels=curriculum",
                "levels.levels=[1-1,1-2]",
                "levels.success_window=2",
                "levels.min_level_episodes=1",
            ],
        ).levels
    cfg.objective = "levels_cleared_score"
    reported = []
    result = train(cfg, overrides=overrides, on_iteration=lambda i, value: reported.append(value) or False)

    # Every episode reaches the flag, so cleared levels count fully
    assert reported[0] == pytest.approx(1.0)
    assert result.objective == reported[-1] >= 1.0
    assert mlflow.get_run(result.run_id).data.metrics["levels_cleared_score"] == pytest.approx(result.objective)


class SavingAgent:
    def __init__(self) -> None:
        self.saved: list[int] = []

    def save(self, save_iter: int) -> None:
        self.saved.append(save_iter)


def test_checkpoint_policy(monkeypatch):
    tags = {}
    monkeypatch.setattr(mlflow, "set_tags", tags.update)
    agent = SavingAgent()
    checkpoints = Checkpoints(agent, save_frequency=10, min_improvement=0.1, min_iters=3)

    # The first score is the best so far, a small improvement does not count, a bigger one only after 3 iterations
    for train_iter, score in [(1, 1.0), (2, 1.05), (3, 1.5), (4, 1.5), (5, 1.0)]:
        checkpoints.after_iteration(train_iter, score, last=False)
    assert agent.saved == [1, 4]
    assert tags == {"best_checkpoint_iter": "4", "best_score": "1.500"}

    # Regular and final checkpoints, without a score
    checkpoints.after_iteration(10, None, last=False)
    checkpoints.after_iteration(12, None, last=True)
    assert agent.saved == [1, 4, 10, 12]


def test_evaluation_config(cfg):
    cfg, overrides = cfg
    run_id = train(cfg, overrides=overrides).run_id
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cli_cfg = compose("neurio_config", overrides=["levels=single", "levels.level=4-2"])

    # By default the full game, with one worker that always takes its most likely action
    config = test_script.evaluation_config(run_id, cli_cfg, levels_overridden=False)
    assert isinstance(config.levels, ProgressionLevelsConfig)
    assert isinstance(config.agent.sampling_strategy, GreedySamplingStrategy)
    assert config.num_workers == 1
    # Otherwise the run's own settings
    assert config.agent.batch_size == 4

    config = test_script.evaluation_config(run_id, cli_cfg, levels_overridden=True)
    assert config.levels.name == "4-2"


def test_load_checkpoint(cfg):
    cfg, overrides = cfg
    run_id = train(cfg, overrides=overrides).run_id
    config = test_script.evaluation_config(run_id, cfg, levels_overridden=True)
    env = train_module.get_multiprocess_environment(1, config.environment, config.levels, None)
    try:
        agent = get_agent(config.agent, get_env_info(env))
        fresh = [p.detach().clone() for p in agent.actor_critic.parameters()]
        agent.load(run_id, test_script.latest_checkpoint_iter(run_id))
    finally:
        env.close()
    trained = mlflow.pytorch.load_state_dict(f"runs:/{run_id}/actor_critic-0003")
    assert all(torch.equal(p, trained[name]) for name, p in agent.actor_critic.state_dict().items())
    assert any(not torch.equal(f, p) for f, p in zip(fresh, agent.actor_critic.parameters(), strict=True))
