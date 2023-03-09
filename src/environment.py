import logging
from typing import Any, Dict, List, Tuple

import cv2
import gym
import gym.logger
from gym.vector import AsyncVectorEnv
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

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
    def __init__(self, config: DictConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_reward_weight = config.score_reward_weight
        self.death_penalty = config.death_penalty
        self.level_finish_reward = config.level_finish_reward

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

        reward += self.score_reward_weight * score_diff

        if terminated or truncated:
            if info["flag_get"]:
                reward += self.level_finish_reward
            else:
                reward -= self.death_penalty

        reward /= 10.0

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
    env = CustomRewardWrapper(config.reward, env)
    env = ActionRepeatWrapper(config.num_repeat_frames, env)
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


def get_stack_frames(env: Env) -> int:
    obs_space = env.observation_space.shape
    assert obs_space is not None
    if len(obs_space) >= 4:
        return obs_space[-4]
    return 1


def get_height(env: Env) -> int:
    obs_space = env.observation_space.shape
    assert obs_space is not None
    assert len(obs_space) >= 3
    return obs_space[-3]


def get_width(env: Env) -> int:
    obs_space = env.observation_space.shape
    assert obs_space is not None
    assert len(obs_space) >= 3
    return obs_space[-2]


def get_channels(env: Env) -> int:
    obs_space = env.observation_space.shape
    assert obs_space is not None
    assert len(obs_space) >= 3
    return obs_space[-1]


def get_num_actions(env: Env) -> int:
    act_space = env.action_space
    if isinstance(act_space, Discrete):
        return act_space.n
    assert isinstance(act_space, MultiDiscrete)
    return act_space.nvec[0]
