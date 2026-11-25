import datetime
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import environment
import evaluation
import train as train_module
from agents import checkpoint_name, get_agent
from environment import EnvironmentInfo, get_env_info, get_environment
from fakes import FakeMarioEnv
from game_loop import GameLoop
from resume import RESUME_ITER_TAG, config_hash, find_previous_run, load_training_state
from test_game_loop import make_vector_env
from train import CONFIG_ARTIFACT, Checkpoints, load_run_config, setup_mlflow, train

ROOT = Path(__file__).resolve().parents[1]


PPO_OVERRIDES = [
    "agent=ppo",
    "agent.batch_size=4",
    # A tiny model keeps the checkpoints small
    "agent.model.channels=[4,4]",
    "agent.model.blocks_per_stage=1",
    "agent.model.fc_size=16",
]
# A tiny model on 16 x 16 frames, trained after the first iteration
DREAMER_OVERRIDES = [
    "environment.height=16",
    "environment.width=16",
    "agent.deter=16",
    "agent.blocks=2",
    "agent.hidden=8",
    "agent.stoch=4",
    "agent.classes=4",
    "agent.depth=2",
    "agent.kernel=3",
    "agent.decoder_blocks=2",
    "agent.units=8",
    "agent.bins=15",
    "agent.batch_size=2",
    "agent.batch_length=4",
    "agent.replay_size=100",
    "agent.imagination_length=2",
    "agent.train_ratio=1",
]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    return _compose(tmp_path, monkeypatch, PPO_OVERRIDES)


@pytest.fixture
def dreamer_cfg(tmp_path, monkeypatch):
    return _compose(tmp_path, monkeypatch, DREAMER_OVERRIDES)


def _compose(tmp_path, monkeypatch, agent_overrides: list[str]):
    # mlflow stores artifacts relative to the working directory
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(environment, "BaseEnvironment", lambda level, **_: FakeMarioEnv([10.0] * 3, level=level))
    monkeypatch.setattr(
        train_module,
        "get_multiprocess_environment",
        lambda num, config, levels, stack_frames, render_mode=None: make_vector_env(
            [get_environment(config, levels, stack_frames) for _ in range(num)]
        ),
    )
    overrides = [
        "levels=single",
        "num_workers=2",
        "num_iters=3",
        "steps_per_iter=4",
        "save_frequency=2",
        "resume_frequency=2",
        # The fake episodes are only a few frames long, too short for random start frames
        "environment.num_repeat_frames=1",
        f"tracking_uri=sqlite:///{tmp_path}/mlflow.db",
        *agent_overrides,
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose("neurio_config", overrides=overrides), overrides


def _artifacts(run_id: str) -> list[str]:
    return sorted(a.path for a in mlflow.MlflowClient().list_artifacts(run_id))


def _cleared(run_id: str) -> float:
    return mlflow.get_run(run_id).data.metrics["levels/cleared"]


def test_train_logs_run(cfg):
    cfg, overrides = cfg
    cfg.tags = {"from_config": "yes"}
    run_id = train(cfg, overrides=overrides)

    run = mlflow.get_run(run_id)
    assert run.data.tags["overrides"] == " ".join(overrides)
    assert run.data.tags["from_config"] == "yes"
    assert "agent.gamma" in run.data.params
    assert {CONFIG_ARTIFACT, checkpoint_name(2), checkpoint_name(3)} <= set(_artifacts(run_id))

    # Stepped by the iteration, with the environment steps of both workers as a metric
    assert _history(run_id, "progress/env_steps") == [(1, 8), (2, 16), (3, 24)]
    # Episodes end at x 30, after 3 steps, without the flag
    assert _history(run_id, "levels/distance_1-1")[-1] == (3, 30.0)
    assert _history(run_id, "levels/cleared")[-1] == (3, 0.0)


def test_stored_config_reproduces_the_run(cfg):
    cfg, overrides = cfg
    run_id = train(cfg, overrides=overrides)

    stored = load_run_config(run_id)
    assert OmegaConf.to_container(stored) == OmegaConf.to_container(cfg, resolve=True)
    assert instantiate(stored, _convert_="all") == instantiate(cfg, _convert_="all")


def _reach_the_flag(monkeypatch):
    monkeypatch.setattr(
        environment, "BaseEnvironment", lambda level, **_: FakeMarioEnv([10.0] * 3, level=level, flag_get=True)
    )


def test_find_best_run(cfg, monkeypatch):
    cfg, overrides = cfg
    first = train(cfg, overrides=overrides)
    _reach_the_flag(monkeypatch)
    second = train(cfg, overrides=overrides)
    assert _cleared(second) > _cleared(first)
    assert evaluation.find_best_run_id(instantiate(cfg, _convert_="all")) == second


def _use_curriculum(cfg, monkeypatch):
    """A curriculum over two levels, on which every episode reaches the flag."""
    _reach_the_flag(monkeypatch)
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


def test_curriculum_keeps_best_checkpoint(cfg, monkeypatch):
    cfg, overrides = cfg
    _use_curriculum(cfg, monkeypatch)
    run_id = train(cfg, overrides=overrides)

    run = mlflow.get_run(run_id)
    # The first iteration already sets the best score, later ones are within the minimum distance between saves
    assert run.data.tags["best_checkpoint_iter"] == "1"
    assert float(run.data.tags["best_score"]) > 0
    assert checkpoint_name(1) in _artifacts(run_id)
    # 1-1 is cleared, 1-2 depends on whether it was sampled after its unlock
    assert _history(run_id, "levels/cleared")[0] == (1, 1.0)
    assert _cleared(run_id) >= 1.0


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

    # Regular and final checkpoints, without a better score
    checkpoints.after_iteration(10, 1.0, last=False)
    checkpoints.after_iteration(12, 1.0, last=True)
    assert agent.saved == [1, 4, 10, 12]


def test_evaluation_config(cfg, monkeypatch):
    cfg, overrides = cfg
    run_id = train(cfg, overrides=overrides)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cli_cfg = compose("neurio_config", overrides=["levels=single", "levels.level=4-2"])

    # The run's own settings and levels, with one worker
    config = evaluation.evaluation_config(run_id, cli_cfg, levels_overridden=False)
    assert config.levels.name == "1-1"
    assert config.num_workers == 1 and config.environment.sticky_action_probability == 0.0
    assert config.agent.batch_size == 4

    config = evaluation.evaluation_config(run_id, cli_cfg, levels_overridden=True)
    assert config.levels.name == "4-2"

    # Runs from before a setting was added get the current one
    stored = load_run_config(run_id)
    del stored.resume_frequency
    monkeypatch.setattr(evaluation, "load_run_config", lambda _: stored)
    assert (
        evaluation.evaluation_config(run_id, cli_cfg, levels_overridden=False).resume_frequency
        == cli_cfg.resume_frequency
    )

    # Settings removed since are dropped, as long as the run had the value the code now always uses
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        curriculum = compose("neurio_config", overrides=["levels=curriculum"]).levels
    stored.levels = OmegaConf.create({**OmegaConf.to_container(curriculum), "unlock_success_rate": 0.02})
    stored.render = True
    assert evaluation.evaluation_config(run_id, cli_cfg, levels_overridden=False).levels.name == "curriculum-32"
    stored.levels.unlock_success_rate = 0.5
    with pytest.raises(AssertionError, match="unlock_success_rate"):
        evaluation.evaluation_config(run_id, cli_cfg, levels_overridden=False)


def test_train_dreamer(dreamer_cfg, tmp_path, monkeypatch):
    cfg, overrides = dreamer_cfg
    run_id = train(cfg, overrides=overrides)

    metrics = mlflow.get_run(run_id).data.metrics
    assert metrics["progress/updates"] > 0 and np.isfinite(metrics["world_model/reconstruction"])
    # Each checkpoint comes with a preview of the world model's predictions
    checkpoint = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id, checkpoint_name(3))]
    assert f"{checkpoint_name(3)}/preview.gif" in checkpoint
    config = evaluation.evaluation_config(run_id, cfg, levels_overridden=True)
    # The fake game has 7 actions
    agent = get_agent(config.agent, EnvironmentInfo(16, 16, 1, 1, 7, 1))
    agent.load(run_id, evaluation.latest_checkpoint_iter(run_id))

    # A better run of another agent is not the best run of this one
    ppo_cfg, ppo_overrides = _compose(tmp_path, monkeypatch, PPO_OVERRIDES)
    _reach_the_flag(monkeypatch)
    assert _cleared(train(ppo_cfg, overrides=ppo_overrides)) > _cleared(run_id)
    assert evaluation.find_best_run_id(instantiate(cfg, _convert_="all")) == run_id


def test_load_checkpoint(cfg, tmp_path):
    cfg, overrides = cfg
    run_id = train(cfg, overrides=overrides)
    config = evaluation.evaluation_config(run_id, cfg, levels_overridden=True)
    env = train_module.get_multiprocess_environment(
        1, config.environment, config.levels, config.agent.stack_frames, None
    )
    try:
        agent = get_agent(config.agent, get_env_info(env))
        fresh = [p.detach().clone() for p in agent.actor_critic.parameters()]
        agent.load(run_id, evaluation.latest_checkpoint_iter(run_id))
    finally:
        env.close()
    path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{checkpoint_name(3)}", dst_path=str(tmp_path))
    trained = torch.load(Path(path) / "networks.pt", weights_only=True)["actor_critic"]
    assert all(torch.equal(p, trained[name]) for name, p in agent.actor_critic.state_dict().items())
    assert any(not torch.equal(f, p) for f, p in zip(fresh, agent.actor_critic.parameters(), strict=True))


class Interrupted(Exception):
    pass


RUN_TRAIN_ITER = GameLoop.run_train_iter


def _interrupt_after(monkeypatch, iteration):
    """Training is interrupted after the given iteration, once its metrics are logged."""

    def interrupted(loop, steps):
        metrics = RUN_TRAIN_ITER(loop, steps)
        if loop.total_iters_finished == iteration:
            raise Interrupted
        return metrics

    monkeypatch.setattr(GameLoop, "run_train_iter", interrupted)


def _history(run_id, key):
    return [
        (m.step, m.value) for m in sorted(mlflow.MlflowClient().get_metric_history(run_id, key), key=lambda m: m.step)
    ]


def test_interrupted_run_resumes_from_last_checkpoint(cfg, monkeypatch):
    cfg, overrides = cfg
    _use_curriculum(cfg, monkeypatch)
    cfg.num_iters = 4
    _interrupt_after(monkeypatch, 3)
    with pytest.raises(Interrupted):
        train(cfg, overrides=overrides)
    monkeypatch.setattr(GameLoop, "run_train_iter", RUN_TRAIN_ITER)
    experiment_id = setup_mlflow(instantiate(cfg, _convert_="all"))
    interrupted = find_previous_run(experiment_id, config_hash(cfg), cfg.num_iters)
    # Saved every 2 iterations, so the third one is repeated
    assert interrupted is not None and interrupted.iteration == 2
    assert interrupted.start_time <= interrupted.last_logged <= datetime.datetime.now()
    _, state = load_training_state(interrupted.run_id)
    assert state["loop"]["level_scheduler"]["num_unlocked"] == 2
    assert state["loop"]["agent"]["self_imitation"]
    assert _history(interrupted.run_id, "progress/env_steps")[-1] == (3, 24)

    run_id = train(cfg, overrides=overrides, resume_run_id=interrupted.run_id)

    assert run_id == interrupted.run_id
    steps = _history(run_id, "progress/env_steps")
    # The repeated third iteration continues from the checkpoint's environment steps, of 2 workers with 4 steps each,
    # and replaces the one logged before the interruption
    assert steps == [(1, 8), (2, 16), (3, 24), (4, 4 * 2 * 4)]
    run = mlflow.get_run(run_id)
    assert run.data.tags[RESUME_ITER_TAG] == "4" and run.data.tags["resumed_after_iter"] == "2"
    # The best checkpoint so far is restored with the state, so the resumed iterations are no new best
    assert run.data.tags["best_checkpoint_iter"] == "1"
    # Finished, so there is nothing left to resume
    finished = find_previous_run(experiment_id, config_hash(cfg), cfg.num_iters)
    assert finished is not None and finished.finished

    # Unless it is continued for more iterations
    cfg.num_iters = 5
    previous = find_previous_run(experiment_id, config_hash(cfg), cfg.num_iters)
    assert previous is not None and not previous.finished
    run_id = train(cfg, overrides=overrides, resume_run_id=interrupted.run_id)
    assert _history(run_id, "progress/env_steps")[-1] == (5, 5 * 2 * 4)


def test_only_runs_with_the_same_config_are_resumable(cfg, monkeypatch):
    cfg, overrides = cfg
    _interrupt_after(monkeypatch, 3)
    with pytest.raises(Interrupted):
        train(cfg, overrides=overrides)
    experiment_id = setup_mlflow(instantiate(cfg, _convert_="all"))

    cfg.tags = {"note": "tags and checkpoint frequency do not matter"}
    cfg.save_frequency = 5
    assert find_previous_run(experiment_id, config_hash(cfg), cfg.num_iters) is not None
    cfg.agent.gamma = 0.5
    assert find_previous_run(experiment_id, config_hash(cfg), cfg.num_iters) is None
