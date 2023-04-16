import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import gym
import gym.logger
from gym.vector import AsyncVectorEnv
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.time_limit import TimeLimit

gym.logger.set_level(logging.ERROR)
import gym_super_mario_bros
import numpy as np
import numpy.typing as npt
from gym.core import Env, ObservationWrapper, Wrapper
from gym.spaces import Box, Discrete, MultiDiscrete
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace  # type: ignore
from omegaconf import DictConfig

BASE_WIDTH = 256
BASE_HEIGHT = 240


class BaseEnvironment(JoypadSpace):
    def __init__(
        self, config: DictConfig, level: str = "", render_mode: str = "human"
    ) -> None:
        name: str = config.env_name
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
    def __init__(self, config: DictConfig, env: Env, **kwargs) -> None:
        super().__init__(env, **kwargs)

        self.clip_top: int = config.clip_top
        self.clip_bot: int = config.clip_bot
        self.clip_left: int = config.clip_left
        self.clip_right: int = config.clip_right

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
        return observation[
            self.clip_top : -self.clip_bot if self.clip_bot else None,
            self.clip_left : -self.clip_right if self.clip_right else None,
            ...,
        ]


class SubsampleWrapper(ObservationWrapper):
    def __init__(self, config: DictConfig, env: Env, **kwargs) -> None:
        super().__init__(env, **kwargs)

        factor: float = config.subsampling_factor

        obs_shape = self.observation_space.shape
        if obs_shape is None:
            obs_shape = (int(BASE_HEIGHT / factor), int(BASE_WIDTH / factor))
        else:
            h, w = obs_shape[:2]
            obs_shape = (int(h / factor), int(w / factor), *obs_shape[2:])

        self.height, self.width = obs_shape[:2]

        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

    def observation(self, observation: npt.NDArray) -> npt.NDArray:
        result = cv2.resize(observation, (self.height, self.width))
        if len(result.shape) == 2:
            result = np.expand_dims(result, -1)
        return result


class ActionRepeatWrapper(Wrapper):
    def __init__(self, num_repeat_frames: int, env: Env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.num_repeat_frames = num_repeat_frames

    def step(
        self, action: int
    ) -> Tuple[npt.NDArray, float, bool, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, metrics = super().step(action)
        total_reward = reward
        for _ in range(self.num_repeat_frames - 1):
            if terminated or truncated:
                break
            state, reward, terminated, truncated, metrics = super().step(action)
            total_reward += reward
        return state, total_reward, terminated, truncated, metrics


class CustomRewardWrapper(Wrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.curr_score = 0

    def reset(self) -> Tuple[npt.NDArray, dict]:
        self.curr_score = 0
        return super().reset()

    def step(
        self, action: int
    ) -> Tuple[npt.NDArray, float, bool, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = super().step(action)

        next_score = info["score"]
        score_diff = next_score - self.curr_score
        self.curr_score = next_score

        reward += score_diff / 40

        if terminated or truncated:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50

        reward /= 10

        return state, reward, terminated, truncated, info


class AddAxisWrapper(Wrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def reset(self) -> Tuple[List[npt.NDArray], dict]:
        state, info = super().reset()
        return [state], {key: [value] for key, value in info.items()}

    def step(
        self, action: List[int]
    ) -> Tuple[List[npt.NDArray], List[float], List[bool], List[bool], Dict[str, Any]]:
        assert len(action) == 1
        state, reward, terminated, truncated, info = super().step(action[0])
        info = {key: [value] for key, value in info.items()}
        return [state], [reward], [terminated], [truncated], info


def get_environment(
    config: DictConfig,
    **kwargs,
) -> Env:
    env = BaseEnvironment(config, **kwargs)
    if config.grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    env = ClipWrapper(config, env)
    env = SubsampleWrapper(config, env)
    env = CustomRewardWrapper(env)
    env = ActionRepeatWrapper(config.num_repeat_frames, env)
    # env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
    env = FrameStack(env, config.num_stack_frames)
    env = RecordEpisodeStatistics(env)
    return env


def get_singleprocess_environment(*args, **kwargs) -> Env:
    env = get_environment(*args, **kwargs)
    return AddAxisWrapper(env)


def get_multiprocess_environment(num_environments: int, *args, **kwargs) -> Env:
    if num_environments <= 1:
        return get_singleprocess_environment(*args, **kwargs)
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
    obs_space = env.observation_space.shape
    assert obs_space is not None
    assert len(obs_space) >= 3

    height, width, channels = obs_space[-3:]
    if len(obs_space) >= 4:
        stack_frames = obs_space[-4]
    else:
        stack_frames = 1

    act_space = env.action_space
    if isinstance(act_space, Discrete):
        num_actions = act_space.n
    else:
        assert isinstance(act_space, MultiDiscrete)
        num_actions = act_space.nvec[0]

    if isinstance(env, AsyncVectorEnv):
        num_workers = env.num_envs
    else:
        num_workers = 1

    return EnvironmentInfo(
        width, height, stack_frames, channels, num_actions, num_workers
    )
