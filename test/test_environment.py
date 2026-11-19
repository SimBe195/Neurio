from typing import ClassVar

import numpy as np
import pytest
from gymnasium.wrappers import GrayscaleObservation

import environment
from config.levels import CurriculumLevelsConfig, RandomLevelsConfig, SingleLevelConfig
from environment import (
    ActionRepeatWrapper,
    BaseEnvironment,
    ClipWrapper,
    FlagRewardWrapper,
    GameViewerWrapper,
    LevelExitTermination,
    LevelSelectEnv,
    NoProgressTruncation,
    Powerup,
    ProgressShaping,
    ResizeWrapper,
    get_base_environment,
    get_env_info,
    get_environment,
    remove_background,
    x_distance,
)
from fakes import FakeMarioEnv, WarpingMarioEnv, make_env_config


@pytest.fixture
def fake_base_env(monkeypatch):
    created = []

    def make(config, level="", **_):
        created.append(FakeMarioEnv(level=level or "1-1"))
        return created[-1]

    monkeypatch.setattr(environment, "BaseEnvironment", make)
    return created


def test_x_distance_takes_the_way_across_the_page_wrap():
    assert x_distance(100, 140) == 40
    assert x_distance(140, 100) == -40
    # A loop of a castle maze sends Mario back four pages from page 3, which wraps to the last pages
    assert x_distance(1016, 65531) == -1021
    assert x_distance(65535, 0) == 1


def test_clip_wrapper_crops_configured_borders():
    config = make_env_config(clip_top=33, clip_bot=19, clip_left=28, clip_right=4)
    env = ClipWrapper(config, FakeMarioEnv())
    obs, _ = env.reset()

    assert obs.shape == env.observation_space.shape == (240 - 33 - 19, 256 - 28 - 4, 3)
    # Columns encode their original index, so the first remaining column must be clip_left
    assert obs[0, 0, 0] == 28
    assert obs[0, -1, 0] == 256 - 4 - 1


@pytest.mark.parametrize("height,width", [(188, 228), (96, 112), (50, 70)])
def test_resize_wrapper_matches_observation_space(height, width):
    # Non-square input to catch swapped width/height
    config = make_env_config(clip_top=33, clip_bot=19, clip_left=28, height=height, width=width)
    env = ResizeWrapper(config, ClipWrapper(config, GrayscaleObservation(FakeMarioEnv(), keep_dim=True)))
    obs, _ = env.reset()

    assert obs.shape == env.observation_space.shape == (height, width, 1)
    # Column values must still increase along the width axis, i.e. the image is not transposed
    assert np.all(np.diff(obs[0, :, 0].astype(int)) >= 0)
    assert obs[0, -1, 0] > obs[0, 0, 0]


def test_remove_background_keeps_blocks_and_sprites():
    ram = np.zeros(0x800, np.uint8)
    # The screen starts 8 pixels into the level's second page, so it shows the end of the block buffer's second half
    # and the start of the first one
    ram[0x071A], ram[0x071C] = 1, 8
    ram[0x0500 + 0xD0 + 2 * 16 + 1] = 0x54  # Row 2, level x 272-287: screen x 8-23, y 64-79
    ram[0x0500 + 12 * 16 + 0] = 0x54  # Row 12, level x 512-527: screen x 248-255, y 224-239
    frame = np.full((240, 256, 3), 9, np.uint8)
    backdrop, background, sprite = 0, 1, 2
    layers = np.full((240, 256), backdrop, np.uint8)
    layers[64:80, 8:24] = layers[224:240, 248:256] = background
    layers[64, 8] = backdrop  # E.g. the sky around a pipe's rim
    layers[150:160, 50:80] = background  # E.g. a bush
    layers[100:108, 200:208] = layers[70, 10] = sprite

    kept = remove_background(frame, layers, ram).any(-1)

    expected = np.zeros((240, 256), bool)
    expected[64:80, 8:24] = expected[224:240, 248:256] = expected[100:108, 200:208] = True
    expected[64, 8] = False
    assert np.array_equal(kept, expected)


def test_action_repeat_sums_rewards_and_stops_at_termination():
    fake = FakeMarioEnv(rewards=[1.0, 2.0, 3.0, 4.0, 5.0])
    env = ActionRepeatWrapper(make_env_config(num_repeat_frames=3), fake)
    env.reset()

    _, reward, terminated, _, _ = env.step(2)
    assert reward == 6.0 and not terminated

    _, reward, terminated, _, _ = env.step(1)
    assert reward == 9.0 and terminated
    assert fake.actions == [2, 2, 2, 1, 1]


def test_sticky_actions_keep_the_previous_frames_action():
    fake = FakeMarioEnv(rewards=[0.0] * 2000)
    env = ActionRepeatWrapper(make_env_config(num_repeat_frames=2, sticky_action_probability=0.5), fake)
    env.reset(seed=0)
    # A different action each step, so that every kept action shows
    chosen = list(range(1, 1001))
    for action in chosen:
        env.step(action)

    # Each frame plays the chosen action or the previous frame's, the first one of the episode no buttons
    per_frame = [action for action in chosen for _ in range(2)]
    previous = [0, *fake.actions[:-1]]
    assert all(played in (action, prev) for played, action, prev in zip(fake.actions, per_frame, previous, strict=True))
    # The first frame of each step keeps the previous action in half the cases, the second if the first did and again
    kept = np.mean([played != action for played, action in zip(fake.actions, per_frame, strict=True)])
    assert kept == pytest.approx((0.5 + 0.25) / 2, abs=0.04)


def test_flag_reward_replaces_the_games():
    env = FlagRewardWrapper(make_env_config(flag_reward=500), FakeMarioEnv(rewards=[1.0, 53.0], flag_get=True))
    env.reset()
    assert env.step(0)[1] == 1.0
    assert env.step(0)[1] == 503.0


def _shaping(env: FakeMarioEnv) -> ProgressShaping:
    # Mario moves 10 pixels per step
    config = make_env_config(reward_scale=0.1, flag_reward=500, death_penalty=100, progress_reward=1.0)
    return ProgressShaping(config, env)


def test_progress_pays_per_pixel():
    env = _shaping(FakeMarioEnv(rewards=[3.0] * 3, truncate=True))
    env.reset()
    rewards, infos = zip(*[(reward, info) for _, reward, _, _, info in (env.step(0) for _ in range(3))], strict=True)
    # A truncated episode keeps its progress
    assert rewards == pytest.approx([1.0] * 3)
    # The game's reward is kept for scoring the episode
    assert [info["game_reward"] for info in infos] == [3.0] * 3


def test_dying_costs_the_penalty():
    env = _shaping(FakeMarioEnv(rewards=[0.0] * 3))
    env.reset()
    assert [env.step(0)[1] for _ in range(3)] == pytest.approx([1.0, 1.0, 1.0 - 10.0])


def test_the_flag_pays_on_top_of_the_progress():
    env = _shaping(FakeMarioEnv(rewards=[0.0] * 2, flag_get=True))
    env.reset()
    assert [env.step(0)[1] for _ in range(2)] == pytest.approx([1.0, 1.0 + 50.0])


class FlagpoleMarioEnv(FakeMarioEnv):
    """Reports the flag from the first step on without ending the episode, like the flagpole slide in the full game."""

    def _info(self, **kwargs) -> dict:
        return {**super()._info(**kwargs), "flag_get": self.t >= 1}


def test_the_flag_pays_once_per_level():
    env = _shaping(FlagpoleMarioEnv(rewards=[0.0] * 3, truncate=True))
    env.reset()
    assert [env.step(0)[1] for _ in range(3)] == pytest.approx([51.0, 1.0, 1.0])


@pytest.mark.parametrize("new_level", [False, True])
def test_changing_area_or_level_is_no_progress(new_level):
    env = _shaping(WarpingMarioEnv(new_level, rewards=[0.0] * 4, truncate=True))
    env.reset()
    assert [env.step(0)[1] for _ in range(4)] == pytest.approx([1.0, 0.0, 1.0, 1.0])


def test_distance_goes_on_in_another_area():
    env = _shaping(WarpingMarioEnv(False, rewards=[0.0] * 4, truncate=True))
    assert env.reset()[1]["distance"] == 0
    # The new area starts at x = 0 again, which is no step back
    assert [env.step(0)[-1]["distance"] for _ in range(4)] == [10, 10, 20, 30]


@pytest.mark.parametrize("new_level", [False, True])
def test_leaving_the_level_ends_the_episode_without_penalty(new_level):
    env = _shaping(LevelExitTermination(WarpingMarioEnv(new_level, rewards=[0.0] * 4, truncate=True)))
    env.reset()
    steps = [env.step(0) for _ in range(2 if new_level else 4)]
    assert [terminated for _, _, terminated, _, _ in steps] == ([False, True] if new_level else [False] * 4)
    assert steps[1][1] == 0.0


class StuckMarioEnv(FakeMarioEnv):
    """Gets to x = 30, then stays there. Changes the area at the steps in `area_changes`."""

    def __init__(self, area_changes: tuple[int, ...] = (), **kwargs) -> None:
        super().__init__(**kwargs)
        self.area_changes = area_changes

    def _info(self, **kwargs) -> dict:
        return {
            **super()._info(**kwargs),
            "x_pos": min(10 * self.t, 30),
            "area": 1 + sum(self.t >= t for t in self.area_changes),
        }


def test_episode_without_progress_is_truncated():
    env = NoProgressTruncation(make_env_config(max_steps_without_progress=4), StuckMarioEnv(rewards=[1.0] * 20))
    for _ in range(2):
        env.reset()
        # Progress up to step 3, then 4 steps without
        truncated = [env.step(0)[3] for _ in range(7)]
        assert truncated == [False] * 6 + [True]


def test_new_area_counts_as_progress():
    env = NoProgressTruncation(
        make_env_config(max_steps_without_progress=4), StuckMarioEnv(rewards=[1.0] * 20, area_changes=(6,))
    )
    env.reset()
    truncated = [env.step(0)[3] for _ in range(10)]
    assert truncated == [False] * 9 + [True]


# Runs of (action of the complex movement, frames) that take Mario from the start of 1-1 down its fourth pipe into the
# room below, and then out of its pipe on the right, found by a random search
INTO_THE_PIPE = [
    *[(2, 20), (1, 10)] * 20,
    (0, 60), (6, 15), (10, 8), (1, 10), (2, 11), (1, 14), (4, 5), (6, 8), (1, 14), (4, 4), (10, 13), (6, 8), (4, 13),
    (1, 16), (2, 11), (10, 7), (4, 12), (10, 15), (6, 5), (1, 14), (0, 21), (2, 9), (0, 15), (6, 13), (1, 7), (4, 7),
    (10, 28), (1, 10), (10, 23), (6, 7), (1, 6), (2, 39), (10, 15), (0, 60),
]  # fmt: skip
OUT_OF_THE_ROOM = [
    (10, 11), (0, 7), (2, 4), (1, 4), (2, 13), (0, 25), (10, 12), (0, 25), (2, 13), (6, 4), (4, 14), (0, 4), (6, 12),
    (10, 6), (1, 31), (4, 13), (1, 17), (0, 11), (2, 11), (6, 8), (4, 15), (6, 15), (4, 12), (10, 11), (0, 10),
]  # fmt: skip


def test_a_pipe_leads_into_another_area_without_a_jump_in_progress():
    config = make_env_config(complex_movement=True)
    env = ProgressShaping(config, BaseEnvironment(config, "1-1"))
    _, start = env.reset()
    steps = [env.step(action) for action, frames in INTO_THE_PIPE + OUT_OF_THE_ROOM for _ in range(frames)]
    infos = [info for *_, info in steps]
    areas = [info["area"] for info in infos]
    room = next(t for t, area in enumerate(areas) if area != start["area"])
    back = areas.index(start["area"], room)
    # Down the pipe at about x 900 into the room, which starts at x 24, and out of it at about x 2600, all in 1-1
    assert infos[room - 1]["x_pos"] > 850 and infos[room]["x_pos"] < 100
    assert infos[back - 1]["x_pos"] < 300 and infos[back]["x_pos"] > 2500
    assert {(info["world"], info["stage"]) for info in infos} == {(1, 1)}
    # Neither is scored as hundreds of pixels back or forth
    assert max(abs(reward) for _, reward, *_ in steps) < 1


def test_full_wrapper_stack(fake_base_env):
    config = make_env_config(clip_top=33, clip_bot=19, clip_left=28, height=96, width=112)
    env = get_environment(config, SingleLevelConfig("1-1"), stack_frames=3)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape == (3, 96, 112, 1)

    info = get_env_info(env)
    assert (info.stack_frames, info.height, info.width, info.image_channels) == (3, 96, 112, 1)
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


def test_levels_start_with_powerups_from_their_distribution(fake_base_env):
    env = LevelSelectEnv(make_env_config(), ["1-1", "2-3"], [0.5, 0.5])
    env.set_start_powerups({"2-3": [0.5, 0.0, 0.5]})
    env.reset(seed=0)
    starts = {"1-1": [], "2-3": []}
    for _ in range(400):
        _, info = env.reset()
        starts[env.current_level].append(info["start_powerup"])
        assert env.step(0)[-1]["start_powerup"] == info["start_powerup"]
    # Levels without a distribution start small
    assert set(starts["1-1"]) == {Powerup.SMALL} and env._envs["1-1"].powerups == []
    assert 0.35 < np.mean(np.array(starts["2-3"]) == Powerup.FIRE) < 0.65
    assert set(env._envs["2-3"].powerups) == {Powerup.FIRE}


def test_powerup_in_the_game():
    """In the real game, a power-up lets Mario survive the first Goomba of 1-1 by shrinking."""
    env = BaseEnvironment(make_env_config(complex_movement=True), level="1-1")
    try:
        # The frames are views of the emulator's screen, so they are copied
        small = env.reset()[0].copy()
        big = env.set_powerup(Powerup.BIG)[0].copy()
        env.reset()
        fire, info = env.set_powerup(Powerup.FIRE)
        assert info["powerup_level"] == Powerup.FIRE
        # Big Mario is drawn taller, fire Mario in other colors
        assert (big != small).any() and (fire != big).any()
        for _ in range(200):
            _, _, terminated, _, info = env.step(1)
            assert not terminated
            if info["powerup_level"] == Powerup.SMALL:
                break
        assert info["powerup_level"] == Powerup.SMALL and info["life"] == 2
    finally:
        env.close()


def test_episodes_start_with_random_frames(fake_base_env):
    env = LevelSelectEnv(make_env_config(num_repeat_frames=4), ["2-3"], [1.0])
    env.reset(seed=0)
    frames = []
    for _ in range(100):
        game = env._envs["2-3"]
        steps_before = len(game.actions)
        env.reset()
        frames.append(len(game.actions) - steps_before)
    assert set(frames) == {0, 1, 2, 3}
    assert len(set(env._envs["2-3"].actions)) > 1


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


@pytest.mark.parametrize(
    "levels", [SingleLevelConfig("3-2"), RandomLevelsConfig(["1-1", "1-2"]), CurriculumLevelsConfig()]
)
def test_base_environment_plays_the_initial_levels(fake_base_env, levels):
    env = get_base_environment(make_env_config(), levels)
    assert isinstance(env, LevelSelectEnv)
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
