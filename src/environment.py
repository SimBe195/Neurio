import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, SupportsFloat

import cv2
import gym_super_mario_bros
import gymnasium
import gymnasium.logger
import numpy as np
import numpy.typing as npt
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from gymnasium.core import Env, ObservationWrapper, RenderFrame, Wrapper
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, VectorEnv
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, RecordEpisodeStatistics, TransformReward
from nes_py.wrappers import JoypadSpace

from config.environment import EnvironmentConfig
from config.levels import LevelConfig, ProgressionLevelsConfig

# gymnasium annotates the level with its default value, so type checkers reject any other
gymnasium.logger.min_level = logging.ERROR  # ty: ignore[invalid-assignment]

BASE_WIDTH = 256
BASE_HEIGHT = 240

# "human" shows the game in a window in real time, "rgb_array" makes frames available through render()
RenderMode = Literal["human", "rgb_array"]


StepType = tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]


class BaseEnvironment(JoypadSpace):
    """
    This class represents the basic game environment, initialized with a certain configuration.
    It inherits from JoypadSpace, which is a class that allows us to translate our own custom actions into sequences
    of button presses that the NES emulator will understand.
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
        name = config.env_name
        if level:
            name = name.replace("Bros-v", f"Bros-{level}-v")
        super().__init__(
            gym_super_mario_bros.make(
                name,
                render_mode=render_mode,
            ),
            COMPLEX_MOVEMENT if config.complex_movement else SIMPLE_MOVEMENT,
        )


class LevelSelectEnv(gymnasium.Env):
    """
    Plays each episode on a level sampled from a configurable distribution. One emulator per level is created lazily
    and kept, since switching the level of an existing emulator is not supported.
    The distribution can be changed during training, e.g. via `AsyncVectorEnv.call("set_level_distribution", ...)`.
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
        self.current_level = levels[0]
        first_env = self._get_env(self.current_level)
        self.observation_space = first_env.observation_space
        self.action_space = first_env.action_space

    def set_level_distribution(self, levels: list[str], weights: list[float]) -> None:
        assert len(levels) == len(weights) > 0
        assert np.isclose(sum(weights), 1.0), "Level weights must sum to 1."
        self.levels = list(levels)
        self.weights = np.asarray(weights, dtype=np.float64)

    def _get_env(self, level: str) -> BaseEnvironment:
        if level not in self._envs:
            self._envs[level] = BaseEnvironment(self.config, level=level, render_mode=self._emulator_render_mode)
        return self._envs[level]

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[npt.NDArray, dict]:
        super().reset(seed=seed)
        self.current_level = self.levels[self.np_random.choice(len(self.levels), p=self.weights)]
        observation, info = self._get_env(self.current_level).reset(seed=seed)
        return observation, info

    def step(self, action: int) -> StepType:
        return self._get_env(self.current_level).step(action)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._get_env(self.current_level).render()

    def close(self) -> None:
        for env in self._envs.values():
            env.close()


class EndOnLifeLossWrapper(Wrapper):
    """
    Ends the episode as soon as Mario dies. Used for the full game, which would otherwise continue until all lives
    are lost.
    """

    def step(self, action: int) -> StepType:
        state, reward, terminated, truncated, info = super().step(action)
        if info["is_dying"] or info["is_dead"]:
            terminated = True
        return state, reward, terminated, truncated, info


def get_base_environment(
    config: EnvironmentConfig,
    levels: LevelConfig,
    render_mode: RenderMode | None = None,
) -> Env:
    """
    Create the unprocessed game environment for the given level configuration.
    """
    if isinstance(levels, ProgressionLevelsConfig):
        return EndOnLifeLossWrapper(BaseEnvironment(config, level="", render_mode=render_mode))
    distribution = levels.initial_distribution()
    assert distribution is not None
    return LevelSelectEnv(config, *distribution, render_mode=render_mode)


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


class SubsampleWrapper(ObservationWrapper):
    """
    This class represents a wrapper for the game environment that subsamples the observations from the game.
    """

    def __init__(self, config: EnvironmentConfig, env: Env, **_) -> None:
        """
        Constructor for the SubsampleWrapper.

        :param config: Environment configuration
        :param env: The environment to be wrapped
        :param kwargs: Additional arguments
        """
        super().__init__(env)

        factor = config.subsampling_factor

        obs_shape = self.observation_space.shape
        if obs_shape is None:
            obs_shape = (int(BASE_HEIGHT / factor), int(BASE_WIDTH / factor))
        else:
            h, w = obs_shape[:2]
            obs_shape = (int(h / factor), int(w / factor), *obs_shape[2:])

        self.height, self.width = obs_shape[:2]
        assert self.height > 0 and self.width > 0, "The sub-sampled observation must have positive dimensions."

        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

    def observation(self, observation: npt.NDArray) -> npt.NDArray:
        """
        Subsample the observation based on the pre-configured settings.

        :param observation: The original observation
        :return: The sub-sampled observation
        """
        result = cv2.resize(observation, (self.width, self.height))
        if len(result.shape) == 2:
            result = np.expand_dims(result, -1)
        return result


class ActionRepeatWrapper(Wrapper):
    """
    This class represents a wrapper for the game environment that repeats the same action for multiple steps.
    """

    def __init__(self, config: EnvironmentConfig, env: Env, **_) -> None:
        """
        Constructor for the ActionRepeatWrapper.

        :param num_repeat_frames: Number of times to repeat the same action
        :param env: The environment to be wrapped
        :param kwargs: Additional arguments
        """
        super().__init__(env)
        self.num_repeat_frames = config.num_repeat_frames

    def step(self, action: int) -> StepType:
        """
        Perform the action and repeat it for a predefined number of frames.
        If the episode terminates or is truncated before all frames are executed, the process is stopped early.

        :param action: The action to be performed
        :return: The state, total reward, termination and truncation flags, and metrics
        """
        state, reward, terminated, truncated, metrics = super().step(action)
        total_reward = float(reward)
        for _ in range(self.num_repeat_frames - 1):
            if terminated or truncated:
                break
            state, reward, terminated, truncated, metrics = super().step(action)
            total_reward += float(reward)
        return state, total_reward, terminated, truncated, metrics


def get_environment(
    config: EnvironmentConfig,
    levels: LevelConfig,
    render_mode: RenderMode | None = None,
) -> Env:
    """
    Set up the game environment with the given configuration.
    This includes several wrappers for preprocessing observations, actions and rewards.

    :param config: Configuration for the environment
    :param levels: Configuration of the levels to play
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
    env = SubsampleWrapper(config, env)
    env = TransformReward(env, lambda reward: float(reward) * config.reward_scale)
    env = ActionRepeatWrapper(config, env)
    env = FrameStackObservation(env, config.num_stack_frames)
    env = RecordEpisodeStatistics(env)
    return env


def get_multiprocess_environment(
    num_environments: int,
    config: EnvironmentConfig,
    levels: LevelConfig,
    render_mode: RenderMode | None = None,
) -> AsyncVectorEnv:
    """The game environments of all workers, each in its own process."""
    return AsyncVectorEnv(
        [lambda: get_environment(config, levels, render_mode) for _ in range(num_environments)],
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
