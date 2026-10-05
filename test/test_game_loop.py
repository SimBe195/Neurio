import mlflow
import numpy as np
import pytest
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

import environment
from agents import Agent, StepOutcome
from config.exploration import ExplorationConfig
from config.levels import CurriculumLevelsConfig, SingleLevelConfig
from environment import get_env_info, get_environment
from fakes import FakeMarioEnv, make_env_config
from game_loop import GameLoop
from level_schedulers import CurriculumLevelScheduler
from novelty import NoveltyBonus


class RecordingAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.observations: list[np.ndarray] = []
        self.outcomes: list[StepOutcome] = []
        self.update_states: list[np.ndarray] = []
        self.update_episode_rewards: list[float | None] = []

    def feed_observation(self, state: np.ndarray) -> None:
        self.observations.append(np.array(state))

    def next_actions(self) -> tuple[list[int], list[float]]:
        return [0] * self.num_workers, [0.0] * self.num_workers

    def give_reward(self, outcome: StepOutcome) -> None:
        self.outcomes.append(outcome)

    def update(self, next_states: np.ndarray, episode_reward: float | None = None) -> dict[str, float]:
        self.update_states.append(np.array(next_states))
        self.update_episode_rewards.append(episode_reward)
        return {"actor_loss": 1.5}


def make_vector_env(envs) -> AsyncVectorEnv:
    # Envs are created in this process and sent to the workers, so that they keep using the fakes
    return AsyncVectorEnv(
        [lambda e=e: e for e in envs], copy=False, context="forkserver", autoreset_mode=AutoresetMode.SAME_STEP
    )


@pytest.fixture
def vector_env(monkeypatch):
    # Worker 0 terminates after 3 steps, worker 1 is truncated after 5 steps
    fake_envs = iter([FakeMarioEnv(rewards=[10.0] * 3), FakeMarioEnv(rewards=[10.0] * 5, truncate=True)])
    monkeypatch.setattr(environment, "BaseEnvironment", lambda config, **_: next(fake_envs))
    env = make_vector_env([get_environment(make_env_config(), SingleLevelConfig()) for _ in range(2)])
    yield env
    env.close()


@pytest.fixture
def logged(monkeypatch):
    calls = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics, step: calls.append((dict(metrics), step)))
    monkeypatch.setattr(mlflow, "log_metric", lambda *args, **kwargs: pytest.fail("Metrics must be logged in batches"))
    return calls


def test_logs_one_batch_per_iteration(vector_env, logged):
    agent = RecordingAgent(None, get_env_info(vector_env))
    loop = GameLoop(vector_env, agent, objective_window=10)
    loop.run_train_iter(steps=5)

    assert len(logged) == 1
    metrics, step = logged[0]
    # Steps of all workers combined
    assert step == 10 == loop.total_env_steps
    assert metrics["iteration"] == 0
    assert metrics["actor_loss"] == 1.5
    # Worker 0 finished an episode of 3 steps, worker 1 one of 5 steps, with rewards of 10 per step scaled by 0.1
    assert metrics["episodes"] == 2
    assert metrics["episode_reward_mean"] == pytest.approx(4.0)
    assert metrics["episode_reward_max"] == pytest.approx(5.0)
    assert metrics["episode_length_mean"] == pytest.approx(4.0)
    assert metrics["episode_x_pos_max"] == 50
    assert metrics["episode_level_index_max"] == 0
    assert metrics["flag_rate"] == 0.0
    assert metrics["recent_mean_reward"] == pytest.approx(4.0) == loop.recent_mean_reward
    assert agent.update_episode_rewards == [pytest.approx(4.0)]

    loop.run_train_iter(steps=5)
    assert logged[1][1] == 20 and logged[1][0]["iteration"] == 1


def test_agent_gets_step_outcomes(vector_env, logged):
    agent = RecordingAgent(None, get_env_info(vector_env))
    loop = GameLoop(vector_env, agent)
    loop.run_train_iter(steps=5)

    # The agent gets the observation after the final step for bootstrapping, not one it already acted on
    assert len(agent.observations) == 5
    assert np.array_equal(agent.update_states[0], loop.current_states)

    ended, truncated = agent.outcomes[2], agent.outcomes[4]
    assert ended.terminated.tolist() == [True, False] and ended.truncated.tolist() == [False, False]
    assert truncated.terminated.tolist() == [False, False] and truncated.truncated.tolist() == [False, True]
    assert ended.final_states[0].shape == truncated.final_states[1].shape == vector_env.single_observation_space.shape
    assert agent.outcomes[3].final_states is None
    assert ended.levels == ["1-1", "1-1"]
    assert ended.bonus is None


def test_recent_mean_reward_window(vector_env, logged):
    loop = GameLoop(vector_env, RecordingAgent(None, get_env_info(vector_env)), objective_window=1)
    loop.run_train_iter(steps=5)
    # Only the last episode (of worker 1, reward 5) counts
    assert loop.recent_mean_reward == pytest.approx(5.0)


def test_without_finished_episodes(vector_env, logged):
    agent = RecordingAgent(None, get_env_info(vector_env))
    loop = GameLoop(vector_env, agent)
    loop.run_train_iter(steps=2)
    assert agent.update_episode_rewards == [None]
    assert loop.recent_mean_reward is None
    assert logged[0][0]["episodes"] == 0
    assert "recent_mean_reward" not in logged[0][0] and "episode_reward_mean" not in logged[0][0]


def test_novelty_bonus_reaches_agent_but_not_episode_metrics(vector_env, logged):
    agent = RecordingAgent(None, get_env_info(vector_env))
    loop = GameLoop(vector_env, agent, novelty_bonus=NoveltyBonus(ExplorationConfig(bonus_scale=0.5)))
    loop.run_train_iter(steps=5)

    # Rewards of 10 per step scaled by 0.1 and, separately, the bonus: in the first step both workers reach the same
    # new cell, so the second one gets 0.5 / sqrt(2)
    first = agent.outcomes[0]
    assert first.rewards == pytest.approx([1.0, 1.0])
    assert first.bonus == pytest.approx([0.5, 0.5 / np.sqrt(2)])
    metrics, _ = logged[0]
    assert metrics["episode_reward_mean"] == pytest.approx(4.0)
    assert metrics["novelty_bonus_per_step"] > 0
    assert metrics["visited_cells"] > 0


def test_curriculum_unlocks_levels_in_workers(monkeypatch, logged):
    # Every episode reaches the flag after 2 steps
    monkeypatch.setattr(
        environment, "BaseEnvironment", lambda config, level, **_: FakeMarioEnv([10.0] * 2, level=level, flag_get=True)
    )
    # A negligible weight for mastered levels, so that all episodes after an unlock are on the new level
    levels = CurriculumLevelsConfig(
        levels=["1-1", "1-2", "1-3"], success_window=2, min_level_episodes=1, min_level_weight=1e-9
    )
    envs = [get_environment(make_env_config(), levels) for _ in range(2)]
    for env in envs:
        for level in levels.levels:
            env.unwrapped._get_env(level)
    vector_env = make_vector_env(envs)
    try:
        scheduler = CurriculumLevelScheduler(levels)
        loop = GameLoop(vector_env, RecordingAgent(None, get_env_info(vector_env)), level_scheduler=scheduler)

        loop.run_train_iter(steps=4)
        assert scheduler.num_unlocked == 2
        assert vector_env.get_attr("levels") == (["1-1", "1-2"], ["1-1", "1-2"])
        assert logged[0][0]["success_rate_1-1"] == 1.0
        assert logged[0][0]["flag_rate"] == 1.0

        # Workers already reset into their next episode before the unlock, the following ones are on the newest level
        loop.run_train_iter(steps=4)
        assert scheduler.num_unlocked == 3
        assert [e.level for e in loop.iter_episodes] == ["1-1", "1-1", "1-2", "1-2"]
        assert [metrics["num_unlocked_levels"] for metrics, _ in logged] == [2, 3]
    finally:
        vector_env.close()


def test_test_episode(monkeypatch):
    monkeypatch.setattr(
        environment, "BaseEnvironment", lambda config, **_: FakeMarioEnv(rewards=[10.0] * 3, flag_get=True)
    )
    env = make_vector_env([get_environment(make_env_config(), SingleLevelConfig())])
    try:
        agent = RecordingAgent(None, get_env_info(env))
        episode = GameLoop(env, agent).run_test_episode()
    finally:
        env.close()
    assert episode.flag_get and episode.x_pos == 30 and episode.level == "1-1"
    assert len(agent.observations) == 3
