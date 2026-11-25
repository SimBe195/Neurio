import mlflow
import numpy as np
import pytest
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

import environment
from agents import Agent, Play, StepOutcome
from config.exploration import ExplorationConfig
from config.levels import CurriculumLevelsConfig, SingleLevelConfig
from environment import LevelSelectEnv, Powerup, get_env_info, get_environment
from evaluation import play_levels, sampled_success_rate
from exploration import NoveltyBonus, UntriedActions
from fakes import FakeMarioEnv, WarpingMarioEnv, make_env_config
from game_loop import GameLoop
from level_schedulers import CurriculumLevelScheduler, LevelScheduler


class RecordingAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.observations: list[np.ndarray] = []
        self.plays: list[Play] = []
        self.resets = 0
        self.untried: list[np.ndarray | None] = []
        self.outcomes: list[StepOutcome] = []
        self.updates = 0
        # Number of actions chosen when each training while playing started
        self.trained_after: list[int] = []

    def act(self, observations: np.ndarray, play: Play = Play.TRAIN, untried: np.ndarray | None = None) -> list[int]:
        self.observations.append(np.array(observations))
        self.plays.append(play)
        self.untried.append(untried)
        return [0] * self.num_workers

    def observe(self, outcome: StepOutcome) -> None:
        # The observations are views of the environments' shared memory
        outcome.observations = np.array(outcome.observations)
        self.outcomes.append(outcome)

    def reset(self) -> None:
        self.resets += 1

    def train_while_playing(self) -> None:
        self.trained_after.append(len(self.observations))

    def update(self) -> dict[str, float]:
        self.updates += 1
        return {"behavior/entropy": 1.5}


def make_vector_env(envs) -> AsyncVectorEnv:
    # Envs are created in this process and sent to the workers, so that they keep using the fakes
    return AsyncVectorEnv(
        [lambda e=e: e for e in envs], copy=False, context="forkserver", autoreset_mode=AutoresetMode.SAME_STEP
    )


@pytest.fixture
def vector_env(monkeypatch):
    # Worker 0 terminates after 3 steps, worker 1 is truncated after 5 steps
    fake_envs = iter([FakeMarioEnv(rewards=[10.0] * 3), FakeMarioEnv(rewards=[10.0] * 5, truncate=True)])
    monkeypatch.setattr(environment, "BaseEnvironment", lambda level, **_: next(fake_envs))
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
    loop = GameLoop(vector_env, agent, LevelScheduler(SingleLevelConfig()))
    loop.run_train_iter(steps=5)

    assert len(logged) == 1
    metrics, step = logged[0]
    # The iteration is the step, the steps of all workers combined a metric
    assert step == 1
    assert metrics["progress/env_steps"] == 10 == loop.total_env_steps
    assert metrics["behavior/entropy"] == 1.5
    # Worker 0 ended at x 30, worker 1 at x 50, neither with the flag
    assert metrics["levels/success_1-1"] == 0.0
    assert metrics["levels/distance_1-1"] == 40.0
    assert metrics["levels/cleared"] == 0.0
    # Trained while the workers played each step, and after the iteration
    assert agent.trained_after == [1, 2, 3, 4, 5] and agent.updates == 1

    loop.run_train_iter(steps=5)
    assert logged[1][1] == 2 and logged[1][0]["progress/env_steps"] == 20


def test_agent_gets_step_outcomes(vector_env, logged):
    agent = RecordingAgent(None, get_env_info(vector_env))
    loop = GameLoop(vector_env, agent)
    loop.run_train_iter(steps=5)

    # Each outcome carries the observations the next actions are chosen for
    assert len(agent.observations) == 5 and set(agent.plays) == {Play.TRAIN}
    for outcome, next_observations in zip(agent.outcomes[:-1], agent.observations[1:], strict=True):
        assert np.array_equal(outcome.observations, next_observations)
    assert np.array_equal(agent.outcomes[-1].observations, loop.current_states)

    ended, truncated = agent.outcomes[2], agent.outcomes[4]
    assert ended.terminated.tolist() == [True, False] and ended.truncated.tolist() == [False, False]
    assert truncated.terminated.tolist() == [False, False] and truncated.truncated.tolist() == [False, True]
    assert ended.final_observations is not None and truncated.final_observations is not None
    assert (
        ended.final_observations[0].shape
        == truncated.final_observations[1].shape
        == vector_env.single_observation_space.shape
    )
    assert agent.outcomes[3].final_observations is None
    assert ended.levels == ["1-1", "1-1"]


def test_agent_learns_with_the_novelty_bonus(vector_env, logged):
    agent = RecordingAgent(None, get_env_info(vector_env))
    novelty = NoveltyBonus(ExplorationConfig(bonus_scale=1.0, cell_size=16))
    loop = GameLoop(vector_env, agent, novelty_bonus=novelty)
    loop.run_train_iter(steps=3)

    # Both workers moved from x 0 to 10 and then to 20, into a new cell each step, with the same reward from the game:
    # the first of them got the full bonus for it and the second the bonus of a second visit
    rewards = np.array([outcome.rewards for outcome in agent.outcomes])
    assert rewards[:2, 0] - rewards[:2, 1] == pytest.approx([1 - 1 / np.sqrt(2)] * 2)
    # The bonus is passed on, so that the agent can tell it from the task's reward
    bonuses = np.array([outcome.bonuses for outcome in agent.outcomes[:2]])
    assert bonuses == pytest.approx(np.array([[1, 1 / np.sqrt(2)]] * 2))
    # How rare each worker's place is, for the agent to replay rare experience more often
    assert [outcome.rarity.tolist() for outcome in agent.outcomes[:2]] == [[1, 0.5], [1, 0.5]]
    # Worker 0's episode ends with the third step at x 30, which counts there, and the next one starts at x 0, which
    # is only looked up: its cell was visited twice, in the first step
    ended = agent.outcomes[2]
    assert ended.rarity.tolist() == [1 / 3, 1 / 4]
    assert ended.first_rarity[0] == 0.5
    assert sorted(novelty.counts.values()) == [2, 4]
    assert logged[0][0]["exploration/visited_cells"] == 2

    restored = NoveltyBonus(ExplorationConfig(bonus_scale=1.0, cell_size=16))
    GameLoop(vector_env, agent, novelty_bonus=restored).load_state_dict(loop.state_dict())
    assert restored.counts == novelty.counts


def test_untried_actions_are_taken_before_a_wall(vector_env, logged):
    agent = RecordingAgent(None, get_env_info(vector_env))
    num_actions = agent.env_info.num_actions
    untried = UntriedActions(cell_size=32, num_actions=num_actions, patience=1)
    loop = GameLoop(vector_env, agent, untried_actions=untried)
    loop.run_train_iter(steps=10)

    # Both workers move 10 pixels per step. Worker 0's episodes end after 3 steps at distance 30, in the first cell,
    # worker 1's after 5 steps at 50, in the second. Worker 0's second episode, ending after step 6, makes no progress,
    # so 1-1 is stuck at the second cell: there, at 40 before step 10, worker 1 takes the actions it has not taken in
    # its first episode
    assert [episode.max_distance for episode in loop.iter_episodes[:2]] == [30, 50]
    masks = [mask for mask in agent.untried if mask is not None]
    assert len(masks) == 10 and not np.any(masks[:9]) and not masks[9][0].any()
    assert masks[9][1].tolist() == [False] + [True] * (num_actions - 1)
    metrics = logged[0][0]
    assert metrics["exploration/untried_actions"] == 1 / 20
    assert metrics["exploration/stuck_levels"] == 1 and metrics["exploration/window_1-1"] == 32

    restored = UntriedActions(cell_size=32, num_actions=num_actions, patience=1)
    GameLoop(vector_env, agent, untried_actions=restored).load_state_dict(loop.state_dict())
    assert restored.tried == untried.tried and restored.walls == untried.walls

    # Evaluation follows the agent's own choices
    GameLoop(vector_env, agent, play=Play.GREEDY).run_single_step()
    assert agent.untried[-1] is None
    for play in (Play.GREEDY, Play.SAMPLE):
        with pytest.raises(AssertionError):
            GameLoop(vector_env, agent, play=play, untried_actions=untried)


def test_without_finished_episodes(vector_env, logged):
    agent = RecordingAgent(None, get_env_info(vector_env))
    loop = GameLoop(vector_env, agent, LevelScheduler(SingleLevelConfig()))
    loop.run_train_iter(steps=2)
    assert agent.updates == 1
    assert "levels/success_1-1" not in logged[0][0] and logged[0][0]["levels/cleared"] == 0.0


def test_episodes_count_for_the_level_they_started_on(monkeypatch):
    # The episode leaves 1-1 through a warp pipe into 1-2 at its second step
    monkeypatch.setattr(environment, "BaseEnvironment", lambda level, **_: WarpingMarioEnv(True, rewards=[0.0] * 4))
    vector_env = make_vector_env([get_environment(make_env_config(), SingleLevelConfig("1-1"))])
    try:
        scheduler = LevelScheduler(SingleLevelConfig("1-1"))
        agent = RecordingAgent(None, get_env_info(vector_env))
        loop = GameLoop(vector_env, agent, level_scheduler=scheduler)
        loop.run_single_step()
        loop.run_single_step()
        assert [episode.level for episode in loop.iter_episodes] == ["1-1"]
        assert agent.outcomes[-1].levels == ["1-1"]
        assert scheduler.success_rate("1-1") == 0.0
    finally:
        vector_env.close()


def test_curriculum_unlocks_levels_in_workers(monkeypatch, logged):
    # Every episode reaches the flag after 2 steps
    monkeypatch.setattr(
        environment, "BaseEnvironment", lambda level, **_: FakeMarioEnv([10.0] * 2, level=level, flag_get=True)
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
        assert logged[0][0]["levels/success_1-1"] == 1.0

        # Workers already reset into their next episode before the unlock, the following ones are on the newest level
        loop.run_train_iter(steps=4)
        assert scheduler.num_unlocked == 3
        assert [e.level for e in loop.iter_episodes] == ["1-1", "1-1", "1-2", "1-2"]
        assert [metrics["levels/unlocked"] for metrics, _ in logged] == [2, 3]
    finally:
        vector_env.close()


def test_test_episode(monkeypatch):
    monkeypatch.setattr(
        environment, "BaseEnvironment", lambda level, **_: FakeMarioEnv(rewards=[10.0] * 3, flag_get=True)
    )
    env = make_vector_env([get_environment(make_env_config(), SingleLevelConfig())])
    try:
        agent = RecordingAgent(None, get_env_info(env))
        episode = GameLoop(env, agent, play=Play.GREEDY).run_test_episode()
    finally:
        env.close()
    assert episode.flag_get and episode.x_pos == 30 and episode.level == "1-1"
    assert len(agent.observations) == 3 and set(agent.plays) == {Play.GREEDY} and not agent.trained_after


def test_sampling_evaluation_does_not_train(vector_env):
    agent = RecordingAgent(None, get_env_info(vector_env))
    loop = GameLoop(vector_env, agent, play=Play.SAMPLE)
    for _ in range(4):
        loop.run_single_step()
    assert set(agent.plays) == {Play.SAMPLE} and not agent.trained_after


def test_restart_starts_new_episodes_of_the_agent(vector_env):
    agent = RecordingAgent(None, get_env_info(vector_env))
    loop = GameLoop(vector_env, agent, play=Play.SAMPLE)
    loop.run_single_step()
    resets = agent.resets
    loop.restart()
    assert agent.resets == resets + 1


def test_restart_plays_the_new_distribution(monkeypatch):
    monkeypatch.setattr(
        environment, "BaseEnvironment", lambda level, **_: FakeMarioEnv([10.0] * 3, level=level, flag_get=True)
    )
    env = make_vector_env([get_environment(make_env_config(), SingleLevelConfig())])
    try:
        loop = GameLoop(env, RecordingAgent(None, get_env_info(env)), play=Play.GREEDY)
        env.call("set_level_distribution", ["1-3"], [1.0])
        loop.restart()
        assert loop.run_test_episode().level == "1-3"
    finally:
        env.close()


def test_levels_are_played_in_order_until_the_first_without_the_flag(monkeypatch):
    cleared = {"1-1", "1-2"}
    monkeypatch.setattr(
        environment,
        "BaseEnvironment",
        lambda level, **_: FakeMarioEnv([10.0] * 3, level=level, flag_get=level in cleared),
    )
    levels = ["1-1", "1-2", "1-3", "1-4"]
    game = get_environment(make_env_config(), SingleLevelConfig())
    # The fakes of all levels are created here, since the workers would create real ones
    level_select = game.unwrapped
    assert isinstance(level_select, LevelSelectEnv)
    for level in levels:
        level_select._get_env(level)
    env = make_vector_env([game])
    try:
        loop = GameLoop(env, RecordingAgent(None, get_env_info(env)), play=Play.GREEDY)
        episodes = play_levels(loop, levels)
    finally:
        env.close()
    assert [(e.level, e.flag_get) for e in episodes] == [("1-1", True), ("1-2", True), ("1-3", False)]


def test_sampled_success_rate_weighs_long_and_short_episodes_alike(monkeypatch):
    # Worker 0 dies after each step, worker 1 reaches the flag after 4 steps
    workers = {0: FakeMarioEnv([10.0], level="1-2"), 1: FakeMarioEnv([10.0] * 4, level="1-2", flag_get=True)}
    games = []
    for worker in workers:
        monkeypatch.setattr(environment, "BaseEnvironment", lambda level, w=worker, **_: workers[w])
        game = get_environment(make_env_config(), SingleLevelConfig())
        level_select = game.unwrapped
        assert isinstance(level_select, LevelSelectEnv)
        level_select._get_env("1-2")
        games.append(game)
    env = make_vector_env(games)
    try:
        loop = GameLoop(env, RecordingAgent(None, get_env_info(env)), play=Play.SAMPLE)
        assert sampled_success_rate(loop, "1-2", episodes=4) == 0.5
    finally:
        env.close()


class FireFlowerMarioEnv(FakeMarioEnv):
    """Collects a fire flower on the way to the flag."""

    def _info(self, **kwargs) -> dict:
        return {**super()._info(**kwargs), "powerup_level": Powerup.FIRE if self.t >= 2 else self.powerup}


def test_levels_keep_the_powerup_they_were_finished_with(monkeypatch):
    def make(level, **_):
        env_class = FireFlowerMarioEnv if level == "1-2" else FakeMarioEnv
        return env_class([10.0] * 3, level=level, flag_get=True)

    monkeypatch.setattr(environment, "BaseEnvironment", make)
    levels = ["1-1", "1-2", "1-3"]
    game = get_environment(make_env_config(), SingleLevelConfig())
    level_select = game.unwrapped
    assert isinstance(level_select, LevelSelectEnv)
    for level in levels:
        level_select._get_env(level)
    env = make_vector_env([game])
    try:
        loop = GameLoop(env, RecordingAgent(None, get_env_info(env)), play=Play.GREEDY)
        episodes = play_levels(loop, levels)
    finally:
        env.close()
    # Mario starts small, finds the flower in 1-2 and keeps it for 1-3
    assert [(e.start_powerup, e.powerup) for e in episodes] == [
        (Powerup.SMALL, Powerup.SMALL),
        (Powerup.SMALL, Powerup.FIRE),
        (Powerup.FIRE, Powerup.FIRE),
    ]


class LoopingMarioEnv(FakeMarioEnv):
    """Gets to x 40 and is then sent back to x 10, like the loops of 8-4's castle."""

    def _info(self, **kwargs) -> dict:
        return {**super()._info(**kwargs), "x_pos": [0, 10, 40, 10][self.t]}


def test_walls_are_where_episodes_got_furthest(monkeypatch):
    monkeypatch.setattr(environment, "BaseEnvironment", lambda level, **_: LoopingMarioEnv(rewards=[0.0] * 3))
    vector_env = make_vector_env([get_environment(make_env_config(), SingleLevelConfig("1-1"))])
    try:
        agent = RecordingAgent(None, get_env_info(vector_env))
        untried = UntriedActions(cell_size=16, num_actions=agent.env_info.num_actions, patience=1)
        loop = GameLoop(vector_env, agent, untried_actions=untried)
        for _ in range(3):
            loop.run_single_step()
    finally:
        vector_env.close()
    (episode,) = loop.iter_episodes
    assert (episode.distance, episode.max_distance) == (10, 40)
    # The furthest cell the episode reached, not the one it ended in
    assert untried.walls["1-1"].best == 40 // 16
