import logging
from typing import Any, Dict, Literal, Optional, Tuple, Union

import cv2
import gym.logger
import gym_super_mario_bros
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from gym.core import Env, ObservationWrapper, Wrapper
from gym.spaces import Box, Discrete, MultiDiscrete
from gym.vector import AsyncVectorEnv
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from config.environment import EnvironmentConfig

gym.logger.set_level(logging.ERROR)

BASE_WIDTH = 256
BASE_HEIGHT = 240


StepType = Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]


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
        render_mode: Optional[Union[Literal["human"], Literal["rgb_array"]]] = None,
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
            gym_super_mario_bros.make(  # type: ignore
                name,
                apply_api_compatibility=True,
                render_mode=render_mode,
            ),
            COMPLEX_MOVEMENT if config.complex_movement else SIMPLE_MOVEMENT,
        )


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
        result = cv2.resize(observation, (self.height, self.width))
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
        total_reward = reward
        for _ in range(self.num_repeat_frames - 1):
            if terminated or truncated:
                break
            state, reward, terminated, truncated, metrics = super().step(action)
            total_reward += reward
        return state, total_reward, terminated, truncated, metrics


class CustomRewardWrapper(Wrapper):
    """
    This class represents a wrapper for the game environment that computes a custom reward based on the score.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor for the CustomRewardWrapper.

        :param args: Positional arguments
        :param kwargs: Additional arguments
        """
        super().__init__(*args, **kwargs)
        self.prev_y_pos = float("inf")
        self.prev_action = -1
        self.fell_off = False

    def reset(self) -> Tuple[npt.NDArray, dict]:
        """
        Reset the environment and the current score.

        :return: The initial state and information
        """
        self.prev_y_pos = float("inf")
        self.prev_action = -1
        self.fell_off = False
        return super().reset()

    def step(self, action: int) -> StepType:
        """
        Perform the action and compute the custom reward.

        :param action: The action to be performed
        :return: The state, custom reward, termination and truncation flags, and info
        """
        state, reward, terminated, truncated, info = super().step(action)

        reward /= 10

        reward += max(0, info["y_pos"] - self.prev_y_pos) / 20
        reward += (self.prev_action == action) / 20

        self.prev_y_pos = info["y_pos"]
        self.prev_action = action

        if info["y_pos"] < 79 and not self.fell_off:
            reward -= 2
            self.fell_off = True

        if terminated or truncated:
            if info["flag_get"]:
                reward += 5
            elif not self.fell_off:
                reward -= 5

        return state, reward, terminated, truncated, info


def get_environment(
    config: EnvironmentConfig,
    **kwargs,
) -> Env:
    """
    Set up the game environment with the given configuration.
    This includes several wrappers for preprocessing observations and actions.

    :param config: Configuration for the environment
    :param kwargs: Additional arguments
    :return: The wrapped environment
    """
    env = BaseEnvironment(config, **kwargs)
    if config.grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    env = ClipWrapper(config, env)
    env = SubsampleWrapper(config, env)
    env = CustomRewardWrapper(env)
    env = ActionRepeatWrapper(config, env)
    env = FrameStack(env, config.num_stack_frames)
    env = RecordEpisodeStatistics(env)
    return env


def get_multiprocess_environment(num_environments: int, *args, **kwargs) -> AsyncVectorEnv:
    """
    Set up a multiprocess game environment.

    :param num_environments: Number of parallel environments
    :param args: Positional arguments for get_environment()
    :param kwargs: Keyword arguments for get_environment()
    :return: The vectorized environment
    """
    return AsyncVectorEnv(
        [lambda: get_environment(*args, **kwargs) for _ in range(num_environments)],
        copy=False,
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


def get_env_info(env: Env) -> EnvironmentInfo:
    """
    Get information about the game environment.

    :param env: The environment to get information from
    :return: An EnvironmentInfo object with details about the environment
    """
    obs_space = env.observation_space.shape
    assert obs_space is not None, "Observation space is None."
    assert len(obs_space) >= 3, "Observation space dimensions are not sufficient."

    height, width, channels = obs_space[-3:]
    if len(obs_space) >= 4:
        stack_frames = obs_space[-4]
    else:
        stack_frames = 1

    act_space = env.action_space
    if isinstance(act_space, Discrete):
        num_actions = act_space.n
    else:
        assert isinstance(act_space, MultiDiscrete), "Action space type not supported."
        num_actions = act_space.nvec[0]

    if isinstance(env, AsyncVectorEnv):
        num_workers = env.num_envs
    else:
        num_workers = 1

    return EnvironmentInfo(width, height, stack_frames, channels, num_actions, num_workers)
