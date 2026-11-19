import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal, SupportsFloat

import cv2
import gym_super_mario_bros
import gymnasium
import gymnasium.logger
import numpy as np
import numpy.typing as npt
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gymnasium.core import Env, ObservationWrapper, RenderFrame, Wrapper
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, VectorEnv
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, RecordEpisodeStatistics, TransformReward
from nes_py.wrappers import JoypadSpace

from config.environment import EnvironmentConfig
from config.levels import LevelConfig

# gymnasium annotates the level with its default value, so type checkers reject any other
gymnasium.logger.min_level = logging.ERROR  # ty: ignore[invalid-assignment]

BASE_WIDTH = 256
BASE_HEIGHT = 240
# The game's x positions: an 8-bit page counter times the page width plus the position within the page
X_RANGE = 256 * BASE_WIDTH

# "human" shows the game in a window in real time, "rgb_array" makes frames available through render()
RenderMode = Literal["human", "rgb_array"]


StepType = tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]


# Where the game keeps its state in the NES's RAM
SCREEN_LEFT_PAGE, SCREEN_LEFT_X = 0x071A, 0x071C
# Address of the data of the part of the level that the game shows (low byte, then high byte), which tells apart every
# part, e.g. the room behind a pipe. The area number that gym-super-mario-bros reports only counts the parts a level is
# made of and stays the same in such rooms. The game loads a part one frame before it moves Mario there.
AREA_DATA = 0x00E7
# Mario's float state: 0 on the ground, e.g. also on a block or a pipe, otherwise jumping, falling or climbing
PLAYER_FLOAT_STATE = 0x001D
# Two blocks of 13 rows x 16 columns of 16 x 16-pixel metatiles, for two screens of 256 pixels of the level, from 32
# pixels below the top of the screen. The game looks up collisions in it, and it holds only what Mario can touch.
BLOCK_BUFFER, BLOCK_BUFFER_SIZE, BLOCK_ROWS, BLOCK_TOP = 0x0500, 0xD0, 13, 32
# The emulator's layer of each screen pixel
BACKGROUND_LAYER, SPRITE_LAYER = 1, 2


class Powerup(IntEnum):
    """Mario's power-up state, as stored in the game's memory."""

    SMALL = 0
    BIG = 1
    FIRE = 2


def remove_background(
    frame: npt.NDArray[np.uint8], layers: npt.NDArray[np.uint8], ram: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """
    Paints everything black that Mario cannot touch, such as the sky, hills, bushes, clouds and the trunks of
    1-3's trees: all but the sprites, e.g. Mario, enemies and moving platforms, and the background within the
    metatiles of the game's block buffer, without the backdrop color between them, e.g. the sky around a pipe's rim.

    :param layers: The emulator's layer of each pixel of the frame
    :param ram: The RAM before the frame was emulated, whose scrolling it shows
    """
    left = int(ram[SCREEN_LEFT_PAGE]) * 256 + int(ram[SCREEN_LEFT_X])
    columns = left + np.arange(BASE_WIDTH)
    cells = BLOCK_BUFFER + columns // 256 % 2 * BLOCK_BUFFER_SIZE + columns % 256 // 16
    blocks = ram[16 * np.arange(BLOCK_ROWS)[:, None] + cells[None, :]] != 0
    keep = layers == SPRITE_LAYER
    keep[BLOCK_TOP:] |= np.repeat(blocks, 16, axis=0)[: BASE_HEIGHT - BLOCK_TOP] & (
        layers[BLOCK_TOP:] == BACKGROUND_LAYER
    )
    return frame * keep[..., None]


def x_distance(before: int, after: int) -> int:
    """Distance between two of the game's x positions, the shorter way across the wrap of its page counter."""
    return (after - before + X_RANGE // 2) % X_RANGE - X_RANGE // 2


def area(ram: npt.NDArray[np.uint8]) -> int:
    """The part of the level, see AREA_DATA."""
    return int(ram[AREA_DATA]) | int(ram[AREA_DATA + 1]) << 8


def on_ground(ram: npt.NDArray[np.uint8]) -> bool:
    """Whether Mario stands on something, see PLAYER_FLOAT_STATE."""
    return int(ram[PLAYER_FLOAT_STATE]) == 0


class BaseEnvironment(JoypadSpace):
    """
    This class represents the basic game environment, initialized with a certain configuration.
    It inherits from JoypadSpace, which is a class that allows us to translate our own custom actions into sequences
    of button presses that the NES emulator will understand.

    info["x_pos"] continues across the wrap of the game's 8-bit page counter: the loops of the castle mazes send Mario
    back four pages, which wraps from the first pages to the last ones and would otherwise appear as a jump of about
    64000 pixels to the right.

    info["area"] tells apart every part of the level, also those behind pipes, see AREA_DATA: Mario's x position in them
    is not comparable with the one where he left. It is the part before the frame, so that it changes together with the
    x position. info["on_ground"] tells whether Mario stands on something after the frame.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        level: str,
        render_mode: RenderMode | None = None,
    ) -> None:
        """
        Constructor for the BaseEnvironment.

        :param config: Configuration for the environment
        :param level: The specific level of the game
        """
        super().__init__(
            gym_super_mario_bros.make(config.env_name.replace("Bros-v", f"Bros-{level}-v"), render_mode=render_mode),
            COMPLEX_MOVEMENT if config.complex_movement else SIMPLE_MOVEMENT,
        )
        self.remove_background = config.remove_background
        # The game's last x position and the continued one
        self._game_x = self._x = 0

    def reset(self, **kwargs) -> tuple[npt.NDArray, dict]:
        observation, info = super().reset(**kwargs)
        if self.remove_background:
            observation = remove_background(observation, self._game.layers, self._game.ram)
        self._game_x = self._x = info["x_pos"]
        return observation, {**info, "area": area(self._game.ram), "on_ground": on_ground(self._game.ram)}

    @property
    def _game(self) -> SuperMarioBrosEnv:
        game = self.unwrapped
        assert isinstance(game, SuperMarioBrosEnv)
        return game

    def step(self, action: int) -> StepType:
        # The frame shows the scrolling that the RAM holds before it is emulated
        ram = np.array(self._game.ram)
        observation, reward, terminated, truncated, info = super().step(action)
        if self.remove_background:
            observation = remove_background(observation, self._game.layers, ram)
        self._x += x_distance(self._game_x, info["x_pos"])
        self._game_x = info["x_pos"]
        info = {**info, "x_pos": self._x, "area": area(ram), "on_ground": on_ground(self._game.ram)}
        return observation, reward, terminated, truncated, info

    # Fire Mario's sprite colors, from the game's table of player colors
    FIRE_COLORS = (0x37, 0x27, 0x16)

    def set_powerup(self, powerup: Powerup) -> tuple[npt.NDArray, dict]:
        """
        Give Mario a power-up at the start of an episode, as if he carried it over from the previous level.

        :return: Observation and info of the next frame, which shows Mario with the power-up
        """
        game = self._game
        game.ram[0x0756] = powerup
        # Mario's size: 0 big, 1 small
        game.ram[0x0754] = int(powerup == Powerup.SMALL)
        if powerup == Powerup.FIRE:
            # The game sets fire Mario's colors when he collects the flower. Queue them in its buffer of video memory
            # updates: the palette address of Mario's sprite colors, their number, the colors and a terminator.
            offset = int(game.ram[0x0300])
            update = [0x3F, 0x11, len(self.FIRE_COLORS), *self.FIRE_COLORS, 0x00]
            game.ram[0x0301 + offset : 0x0301 + offset + len(update)] = update
            game.ram[0x0300] = offset + len(update) - 1
        # The game's reward for power-ups compares with the last state, which would reward this one
        game._status_last = game._powerup_level
        # The second frame shows Mario with the power-up
        for _ in range(2):
            observation, _, _, _, info = self.step(0)
        return observation, info


class LevelSelectEnv(gymnasium.Env):
    """
    Plays each episode on a level sampled from a configurable distribution. One emulator per level is created lazily
    and kept, since switching the level of an existing emulator is not supported.
    The distribution can be changed during training, e.g. via `AsyncVectorEnv.call("set_level_distribution", ...)`.

    Mario starts small, or with a power-up sampled per level from the distribution set with `set_start_powerups`, as
    if carried over from the previous level. Steps report it as info["start_powerup"], and the level the episode started
    on as info["level"], which differs from the game's world and stage after a warp pipe.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        levels: list[str],
        weights: list[float],
        render_mode: RenderMode | None = None,
    ) -> None:
        self.config = config
        self.render_mode = render_mode
        self._emulator_render_mode = render_mode
        self._envs: dict[str, BaseEnvironment] = {}
        self.set_level_distribution(levels, weights)
        self.start_powerups: dict[str, list[float]] = {}
        self.start_powerup = Powerup.SMALL
        self.current_level = levels[0]
        first_env = self._get_env(self.current_level)
        self.observation_space = first_env.observation_space
        self.action_space = first_env.action_space

    def set_level_distribution(self, levels: list[str], weights: list[float]) -> None:
        assert len(levels) == len(weights) > 0
        assert np.isclose(sum(weights), 1.0), "Level weights must sum to 1."
        self.levels = list(levels)
        self.weights = np.asarray(weights, dtype=np.float64)

    def set_start_powerups(self, distributions: dict[str, list[float]]) -> None:
        """:param distributions: Per level, the probabilities of starting small, big and with the fire flower"""
        assert all(len(probs) == len(Powerup) and np.isclose(sum(probs), 1.0) for probs in distributions.values())
        self.start_powerups = dict(distributions)

    def _get_env(self, level: str) -> BaseEnvironment:
        if level not in self._envs:
            self._envs[level] = BaseEnvironment(self.config, level=level, render_mode=self._emulator_render_mode)
        return self._envs[level]

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[npt.NDArray, dict]:
        super().reset(seed=seed)
        self.current_level = self.levels[self.np_random.choice(len(self.levels), p=self.weights)]
        env = self._get_env(self.current_level)
        observation, info = env.reset(seed=seed)
        probs = self.start_powerups.get(self.current_level, [1.0, 0.0, 0.0])
        self.start_powerup = Powerup(self.np_random.choice(len(Powerup), p=probs))
        if self.start_powerup != Powerup.SMALL:
            observation, info = env.set_powerup(self.start_powerup)
        # A random action for up to one less frame than an action is repeated shifts the timing of the episode against
        # the repeated actions and the enemies, so that the agent cannot rely on one exact course of the level
        assert isinstance(env.action_space, Discrete)
        for _ in range(self.np_random.integers(self.config.num_repeat_frames)):
            observation, _, _, _, info = env.step(int(self.np_random.integers(env.action_space.n)))
        return observation, self._info(info)

    def step(self, action: int) -> StepType:
        observation, reward, terminated, truncated, info = self._get_env(self.current_level).step(action)
        return observation, reward, terminated, truncated, self._info(info)

    def _info(self, info: dict[str, Any]) -> dict[str, Any]:
        return {**info, "level": self.current_level, "start_powerup": int(self.start_powerup)}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._get_env(self.current_level).render()

    def close(self) -> None:
        for env in self._envs.values():
            env.close()


def get_base_environment(
    config: EnvironmentConfig,
    levels: LevelConfig,
    render_mode: RenderMode | None = None,
) -> Env:
    """Create the unprocessed game environment for the given level configuration."""
    return LevelSelectEnv(config, *levels.initial_distribution(), render_mode=render_mode)


class GameViewerWrapper(Wrapper):
    """
    Shows the game in a window at the NES frame rate, for watching the agent play. Wraps the unprocessed game, which
    must use render mode "rgb_array", so that every emulated frame is shown in one window for all levels.
    """

    FRAMES_PER_SECOND = 60

    def __init__(self, env: Env, scale: int = 3) -> None:
        super().__init__(env)
        self.scale = scale
        self.viewer = None
        self._next_frame_time = 0.0

    def _show(self) -> None:
        if self.viewer is None:
            # Imported here since it needs a display
            from nes_py._image_viewer import ImageViewer

            self.viewer = ImageViewer("Neurio", height=BASE_HEIGHT * self.scale, width=BASE_WIDTH * self.scale)
        self.viewer.show(self.env.render())
        # Wait for the frame's turn, without catching up on frames that were late
        now = time.perf_counter()
        if self._next_frame_time > now:
            time.sleep(self._next_frame_time - now)
        self._next_frame_time = max(now, self._next_frame_time) + 1 / self.FRAMES_PER_SECOND

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[npt.NDArray, dict]:
        result = super().reset(seed=seed, options=options)
        self._show()
        return result

    def step(self, action: int) -> StepType:
        result = super().step(action)
        self._show()
        return result

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        super().close()


class ClipWrapper(ObservationWrapper):
    """
    This class represents a wrapper for the game environment that clips the observations from the game to a certain size
    """

    def __init__(self, config: EnvironmentConfig, env: Env, **_) -> None:
        """
        Constructor for the ClipWrapper.

        :param config: Environment configuration
        :param env: The environment to be wrapped
        :param kwargs: Additional arguments
        """
        super().__init__(env)

        self.clip_top = config.clip_top
        self.clip_bot = config.clip_bot
        self.clip_left = config.clip_left
        self.clip_right = config.clip_right

        self.new_width = BASE_WIDTH - self.clip_left - self.clip_right
        self.new_height = BASE_HEIGHT - self.clip_top - self.clip_bot

        obs_shape = self.observation_space.shape
        if obs_shape is None:
            obs_shape = (self.new_height, self.new_width)
        else:
            obs_shape = (self.new_height, self.new_width, *obs_shape[2:])

        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

    def observation(self, observation: npt.NDArray) -> npt.NDArray:
        """
        Clip the observation based on the pre-configured settings.

        :param observation: The original observation
        :return: The clipped observation
        """
        return observation[
            self.clip_top : -self.clip_bot if self.clip_bot else None,
            self.clip_left : -self.clip_right if self.clip_right else None,
            ...,
        ]


class ResizeWrapper(ObservationWrapper):
    """Resizes the observations to the configured height and width."""

    def __init__(self, config: EnvironmentConfig, env: Env) -> None:
        super().__init__(env)
        self.height, self.width = config.height, config.width
        shape = self.observation_space.shape
        assert shape is not None
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, *shape[2:]), dtype=np.uint8)

    def observation(self, observation: npt.NDArray) -> npt.NDArray:
        result = cv2.resize(observation, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if result.ndim == 2:
            result = np.expand_dims(result, -1)
        return result


class FlagRewardWrapper(Wrapper):
    """Replaces the game's reward for reaching the flag with the configured one, for the episode scores."""

    def __init__(self, config: EnvironmentConfig, env: Env) -> None:
        super().__init__(env)
        self.flag_reward = config.flag_reward

    def step(self, action: int) -> StepType:
        state, reward, terminated, truncated, info = super().step(action)
        completion = info["reward_components"]["completion"]
        if completion:
            reward = float(reward) - completion + self.flag_reward
        return state, reward, terminated, truncated, info


class ActionRepeatWrapper(Wrapper):
    """
    Plays each action for `num_repeat_frames` frames and sums their rewards, stopping early when the episode ends. With
    sticky actions, each frame keeps the previous frame's action with `sticky_action_probability`, as in the Atari
    benchmark (Machado et al., 2018), so that an action takes effect a few frames late now and then.
    """

    # The action before the first frame of an episode: no buttons pressed
    NOOP = 0

    def __init__(self, config: EnvironmentConfig, env: Env) -> None:
        super().__init__(env)
        self.num_repeat_frames = config.num_repeat_frames
        self.sticky_action_probability = config.sticky_action_probability
        self._action = self.NOOP

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        self._action = self.NOOP
        return super().reset(**kwargs)

    def step(self, action: int) -> StepType:
        total_reward = 0.0
        for _ in range(self.num_repeat_frames):
            if self.np_random.random() >= self.sticky_action_probability:
                self._action = action
            state, reward, terminated, truncated, info = super().step(self._action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return state, total_reward, terminated, truncated, info


class NoProgressTruncation(Wrapper):
    """
    Truncates the episode once Mario has not got further to the right for a number of steps, so that dead ends where
    he can only wait for the time to run out take up little of the training. Truncated rather than ended as a death,
    since waiting is part of the game: the agent learns from the value of the last state as if the episode went on.
    Entering another area, e.g. through a pipe, counts as progress.
    """

    def __init__(self, config: EnvironmentConfig, env: Env) -> None:
        super().__init__(env)
        assert config.max_steps_without_progress is not None
        self.max_steps = config.max_steps_without_progress
        self._area: tuple[int, int, int] | None = None
        self._max_x = 0
        self._steps = 0

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        observation, info = super().reset(**kwargs)
        self._area = None
        return observation, info

    def step(self, action: int) -> StepType:
        state, reward, terminated, truncated, info = super().step(action)
        area = (info["world"], info["stage"], info["area"])
        if area != self._area or info["x_pos"] > self._max_x:
            self._area, self._max_x, self._steps = area, info["x_pos"], 0
        else:
            self._steps += 1
        return state, reward, terminated, truncated or self._steps >= self.max_steps, info


class LevelExitTermination(Wrapper):
    """
    Ends the episode when Mario leaves the level other than by its flag, i.e. through a warp pipe, which the game
    otherwise continues in the level it leads to. Ended rather than truncated, so that the levels beyond are worth
    nothing to the agent and the level's own flag is always the better way out. ProgressShaping neither rewards nor
    penalizes it, and the level counts as not cleared.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._level: tuple[int, int] = (0, 0)

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        observation, info = super().reset(**kwargs)
        self._level = (info["world"], info["stage"])
        return observation, info

    def step(self, action: int) -> StepType:
        state, reward, terminated, truncated, info = super().step(action)
        left = (info["world"], info["stage"]) != self._level and not info["flag_get"]
        return state, reward, terminated or left, truncated, info


class ProgressShaping(Wrapper):
    """
    The reward the agent learns from: `progress_reward` per pixel to the right, `flag_reward` for reaching the flag and
    `death_penalty` for dying. Against the game's reward, progress is worth less than dying costs, so that running into
    a pit is no longer worth the distance gained while falling. Changes of the area, e.g. through a pipe or to the next
    level, are not counted as progress, and an episode ended by leaving the level is no death, see
    LevelExitTermination.

    info["distance"] is the progress of the episode so far as it is rewarded: the changes of x within each area. Unlike
    the x position, it does not drop when Mario gets further through a pipe into another area.

    The wrapped environment's reward, the game's, is passed on as info["game_reward"]: it scores the episodes.
    """

    def __init__(self, config: EnvironmentConfig, env: Env) -> None:
        super().__init__(env)
        self.scale = config.reward_scale
        self.flag_reward = config.flag_reward
        self.death_penalty = config.death_penalty
        self.progress_reward = config.progress_reward
        self._place: tuple[int, int, int] = (0, 0, 0)
        self._x = 0
        self._flags: set[tuple[int, int]] = set()
        self._distance = 0

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        observation, info = super().reset(**kwargs)
        self._place, self._x = (info["world"], info["stage"], info["area"]), info["x_pos"]
        self._flags = set()
        self._distance = 0
        return observation, {**info, "distance": 0}

    def step(self, action: int) -> StepType:
        state, game_reward, terminated, truncated, info = super().step(action)
        place = (info["world"], info["stage"], info["area"])
        progress = info["x_pos"] - self._x if place == self._place else 0
        self._distance += progress
        reward = self.progress_reward * progress
        left_level = place[:2] != self._place[:2]
        self._place, self._x = place, info["x_pos"]
        # The flag counts once per level, the game reports it during the whole flagpole slide
        if info["flag_get"] and place[:2] not in self._flags:
            self._flags.add(place[:2])
            reward += self.flag_reward
        elif terminated and not info["flag_get"] and not left_level:
            reward -= self.death_penalty
        info = {**info, "game_reward": float(game_reward), "distance": self._distance}
        return state, reward * self.scale, terminated, truncated, info


def get_environment(
    config: EnvironmentConfig,
    levels: LevelConfig,
    stack_frames: int = 1,
    render_mode: RenderMode | None = None,
) -> Env:
    """
    Set up the game environment with the given configuration.
    This includes several wrappers for preprocessing observations, actions and rewards.

    :param config: Configuration for the environment
    :param levels: Configuration of the levels to play
    :param stack_frames: Number of most recent frames in each observation, see AgentConfig.stack_frames
    :param render_mode: See RenderMode
    :return: The wrapped environment
    """
    if render_mode == "human":
        env = GameViewerWrapper(get_base_environment(config, levels, render_mode="rgb_array"))
    else:
        env = get_base_environment(config, levels, render_mode=render_mode)
    if config.grayscale:
        env = GrayscaleObservation(env, keep_dim=True)
    env = ClipWrapper(config, env)
    env = ResizeWrapper(config, env)
    env = FlagRewardWrapper(config, env)
    env = TransformReward(env, lambda reward: float(reward) * config.reward_scale)
    env = ActionRepeatWrapper(config, env)
    env = LevelExitTermination(env)
    if config.max_steps_without_progress is not None:
        env = NoProgressTruncation(config, env)
    # The episode statistics sum the game's reward, the agent learns from the shaped one
    env = RecordEpisodeStatistics(env)
    env = ProgressShaping(config, env)
    return FrameStackObservation(env, stack_frames)


def get_multiprocess_environment(
    num_environments: int,
    config: EnvironmentConfig,
    levels: LevelConfig,
    stack_frames: int = 1,
    render_mode: RenderMode | None = None,
) -> AsyncVectorEnv:
    """The game environments of all workers, each in its own process."""
    return AsyncVectorEnv(
        [lambda: get_environment(config, levels, stack_frames, render_mode) for _ in range(num_environments)],
        copy=False,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


@dataclass
class EnvironmentInfo:
    width: int
    height: int
    stack_frames: int
    image_channels: int
    num_actions: int
    num_workers: int

    @property
    def total_channel_dim(self):
        return self.stack_frames * self.image_channels


def get_env_info(env: Env | VectorEnv) -> EnvironmentInfo:
    """
    Get information about the game environment.

    :param env: The environment to get information from
    :return: An EnvironmentInfo object with details about the environment
    """
    obs_space = env.observation_space.shape
    assert obs_space is not None, "Observation space is None."
    assert len(obs_space) >= 3, "Observation space dimensions are not sufficient."

    height, width, channels = obs_space[-3:]
    stack_frames = obs_space[-4] if len(obs_space) >= 4 else 1

    act_space = env.action_space
    if isinstance(act_space, Discrete):
        num_actions = int(act_space.n)
    else:
        assert isinstance(act_space, MultiDiscrete), "Action space type not supported."
        num_actions = int(act_space.nvec[0])

    num_workers = env.num_envs if isinstance(env, AsyncVectorEnv) else 1

    return EnvironmentInfo(width, height, stack_frames, channels, num_actions, num_workers)
