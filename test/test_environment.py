from typing import ClassVar

import numpy as np
import pytest
from gymnasium.wrappers import GrayscaleObservation

import environment
from config.levels import CurriculumLevelsConfig, ProgressionLevelsConfig, RandomLevelsConfig, SingleLevelConfig
from environment import (
    ActionRepeatWrapper,
    ClipWrapper,
    EndOnLifeLossWrapper,
    GameViewerWrapper,
    LevelSelectEnv,
    SubsampleWrapper,
    get_base_environment,
    get_env_info,
    get_environment,
)
from fakes import FakeMarioEnv, make_env_config


@pytest.fixture
def fake_base_env(monkeypatch):
    created = []

    def make(config, level="", **_):
        created.append(FakeMarioEnv(level=level or "1-1"))
        return created[-1]

    monkeypatch.setattr(environment, "BaseEnvironment", make)
    return created


def test_clip_wrapper_crops_configured_borders():
    config = make_env_config(clip_top=33, clip_bot=19, clip_left=28, clip_right=4)
    env = ClipWrapper(config, FakeMarioEnv())
    obs, _ = env.reset()

    assert obs.shape == env.observation_space.shape == (240 - 33 - 19, 256 - 28 - 4, 3)
    # Columns encode their original index, so the first remaining column must be clip_left
    assert obs[0, 0, 0] == 28
    assert obs[0, -1, 0] == 256 - 4 - 1


@pytest.mark.parametrize("factor", [1, 2, 3.5])
def test_subsample_wrapper_matches_observation_space(factor):
    # Non-square input to catch swapped width/height
    config = make_env_config(clip_top=33, clip_bot=19, clip_left=28, subsampling_factor=factor)
    env = SubsampleWrapper(config, ClipWrapper(config, GrayscaleObservation(FakeMarioEnv(), keep_dim=True)))
    obs, _ = env.reset()

    assert obs.shape == env.observation_space.shape
    # Column values must still increase along the width axis, i.e. the image is not transposed
    assert np.all(np.diff(obs[0, :, 0].astype(int)) >= 0)
    assert obs[0, -1, 0] > obs[0, 0, 0]


def test_action_repeat_sums_rewards_and_stops_at_termination():
    fake = FakeMarioEnv(rewards=[1.0, 2.0, 3.0, 4.0, 5.0])
    env = ActionRepeatWrapper(make_env_config(num_repeat_frames=3), fake)
    env.reset()

    _, reward, terminated, _, _ = env.step(2)
    assert reward == 6.0 and not terminated

    _, reward, terminated, _, _ = env.step(1)
    assert reward == 9.0 and terminated
    assert fake.actions == [2, 2, 2, 1, 1]


def test_full_wrapper_stack(fake_base_env):
    config = make_env_config(clip_top=33, clip_bot=19, clip_left=28, subsampling_factor=2, num_stack_frames=3)
    env = get_environment(config, SingleLevelConfig("1-1"))
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape == (3, 94, 114, 1)

    info = get_env_info(env)
    assert (info.stack_frames, info.height, info.width, info.image_channels) == (3, 94, 114, 1)
    assert info.total_channel_dim == 3
    assert info.num_workers == 1

    terminated = False
    while not terminated:
        obs, _, terminated, _, info = env.step(0)
    assert obs.shape == env.observation_space.shape
    assert info["episode"]["l"] == 5
    # The game's reward is only scaled
    assert info["episode"]["r"] == pytest.approx(0.5)


def test_level_select_env_samples_from_distribution(fake_base_env):
    env = LevelSelectEnv(make_env_config(), ["1-1", "2-3"], [0.25, 0.75])
    env.reset(seed=0)
    counts = {"1-1": 0, "2-3": 0}
    for _ in range(400):
        _, info = env.reset()
        counts[env.current_level] += 1
        assert f"{info['world']}-{info['stage']}" == env.current_level
    assert 0.65 < counts["2-3"] / 400 < 0.85
    # One emulator per level, created once and reused
    assert len(fake_base_env) == 2


def test_level_select_env_distribution_can_change(fake_base_env):
    env = LevelSelectEnv(make_env_config(), ["1-1"], [1.0])
    env.reset(seed=0)
    env.set_level_distribution(["1-1", "1-2"], [0.0, 1.0])
    env.reset()
    assert env.current_level == "1-2"
    env.step(0)
    assert env._envs["1-2"].actions == [0]

    with pytest.raises(AssertionError):
        env.set_level_distribution(["1-1", "1-2"], [0.5, 0.6])


def test_end_on_life_loss():
    env = EndOnLifeLossWrapper(FakeMarioEnv(rewards=[1.0] * 10, dying_at=3))
    env.reset()
    assert not env.step(0)[2]
    assert not env.step(0)[2]
    assert env.step(0)[2]


@pytest.mark.parametrize(
    "levels,expected_type",
    [
        (SingleLevelConfig("3-2"), LevelSelectEnv),
        (RandomLevelsConfig(["1-1", "1-2"]), LevelSelectEnv),
        (CurriculumLevelsConfig(), LevelSelectEnv),
        (ProgressionLevelsConfig(), EndOnLifeLossWrapper),
    ],
)
def test_base_environment_per_level_config(fake_base_env, levels, expected_type):
    env = get_base_environment(make_env_config(), levels)
    assert isinstance(env, expected_type)
    if isinstance(levels, ProgressionLevelsConfig):
        # The full game
        assert fake_base_env[0].world == 1 and fake_base_env[0].stage == 1
    else:
        assert env.levels == levels.initial_distribution()[0]


class FakeViewer:
    instances: ClassVar[list["FakeViewer"]] = []

    def __init__(self, caption, height, width):
        self.size = (height, width)
        self.frames = 0
        self.closed = False
        FakeViewer.instances.append(self)

    def show(self, frame):
        assert frame.shape == (240, 256, 3)
        self.frames += 1

    def close(self):
        self.closed = True


def test_human_render_mode_shows_every_frame(fake_base_env, monkeypatch):
    import nes_py._image_viewer

    FakeViewer.instances = []
    monkeypatch.setattr(nes_py._image_viewer, "ImageViewer", FakeViewer)
    # Do not wait for real time in the test
    monkeypatch.setattr(GameViewerWrapper, "FRAMES_PER_SECOND", 1e9)
    env = get_environment(make_env_config(num_repeat_frames=2), SingleLevelConfig("1-1"), render_mode="human")

    env.reset()
    env.step(0)
    env.step(0)
    # One window, showing the frame after the reset and both frames of each repeated action
    assert len(FakeViewer.instances) == 1
    viewer = FakeViewer.instances[0]
    assert viewer.frames == 1 + 2 * 2
    assert viewer.size == (720, 768)
    env.close()
    assert viewer.closed


def test_viewer_paces_frames(monkeypatch):
    import nes_py._image_viewer

    monkeypatch.setattr(nes_py._image_viewer, "ImageViewer", FakeViewer)
    clock = {"now": 100.0}
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr("environment.time.perf_counter", lambda: clock["now"])
    monkeypatch.setattr("environment.time.sleep", sleep)
    env = GameViewerWrapper(FakeMarioEnv())
    env.reset()
    # Emulation is instant here, so every later frame waits for its turn
    env.step(0)
    env.step(0)
    assert sleeps == pytest.approx([1 / 60, 1 / 60])

    # A frame that comes late is shown at once, and the schedule does not try to catch up
    clock["now"] += 1.0
    env.step(0)
    assert len(sleeps) == 2
    env.step(0)
    assert sleeps[-1] == pytest.approx(1 / 60)
