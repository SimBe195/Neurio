import logging
from typing import Any, Dict, List, Tuple

import gym
from gym.vector import AsyncVectorEnv
from gym.wrappers import (
    FrameStack,
    GrayScaleObservation,
    RecordEpisodeStatistics,
    RecordVideo,
)

gym.logger.set_level(logging.ERROR)
import gym_super_mario_bros
import numpy as np
from gym.core import Env, ObservationWrapper, Wrapper
from gym.spaces import Box
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from omegaconf import DictConfig


def get_num_actions(config: DictConfig) -> int:
    return len(COMPLEX_MOVEMENT) if config.complex_movement else len(SIMPLE_MOVEMENT)


class BaseEnvironment(JoypadSpace):
    def __init__(
        self, config: DictConfig, level: str = "", render_mode="human"
    ) -> None:
        name = config.env_name
        if level:
            name = name.replace("Bros-v", f"Bros-{level}-v")
        super().__init__(
            gym_super_mario_bros.make(
                name,
                apply_api_compatibility=True,
                render_mode=render_mode,
            ),
            COMPLEX_MOVEMENT if config.complex_movement else SIMPLE_MOVEMENT,
        )


class ClipWrapper(ObservationWrapper):
    def __init__(self, config: DictConfig, env: Env, **kwargs) -> None:
        super().__init__(env, **kwargs)

        self.clip_top = config.clip_top
        self.clip_bot = config.clip_bot
        self.clip_left = config.clip_left
        self.clip_right = config.clip_right

        self.new_width = 256 - self.clip_left - self.clip_right
        self.new_height = 240 - self.clip_top - self.clip_bot

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(
                self.new_height,
                self.new_width,
                *(self.observation_space.shape[2:]),
            ),
            dtype=np.uint8,
        )

    def observation(self, observation: np.array) -> np.array:
        return observation[
            self.clip_top : -self.clip_bot if self.clip_bot else None,
            self.clip_left : -self.clip_right if self.clip_right else None,
            ...,
        ]


class FrameSkipWrapper(Wrapper):
    def __init__(self, num_skip_frames: int, env: Env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.num_skip_frames = num_skip_frames

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        total_reward = 0
        for _ in range(self.num_skip_frames):
            state, reward, terminated, truncated, metrics = super().step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return state, total_reward, terminated, truncated, metrics


class CustomRewardWrapper(Wrapper):
    def __init__(self, config: DictConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_reward_weight = config.score_reward_weight
        self.death_penalty = config.death_penalty
        self.level_finish_reward = config.level_finish_reward

        self.curr_score = 0

    def reset(self) -> np.array:
        self.curr_score = 0
        return super().reset()

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict[str, Any]]:
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

    def reset(self) -> np.array:
        state, info = super().reset()
        return [state], {key: [value] for key, value in info.items()}

    def step(self, action: List[int]) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = super().step(action[0])
        info = {key: [value] for key, value in info.items()}
        return [state], [reward], [terminated], [truncated], info


def get_environment(
    config: DictConfig,
    recording_path: str = "",
    video_prefix: str = "",
    **kwargs,
) -> BaseEnvironment:
    env = BaseEnvironment(config, **kwargs)
    if recording_path:
        env = RecordVideo(
            env,
            recording_path,
            name_prefix=video_prefix,
            episode_trigger=lambda _: True,
        )
    if config.grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    env = ClipWrapper(config, env)
    env = CustomRewardWrapper(config.reward, env)
    env = FrameSkipWrapper(config.num_skip_frames, env)
    env = FrameStack(env, config.num_stack_frames)
    env = RecordEpisodeStatistics(env)
    return env


def get_singleprocess_environment(*args, **kwargs) -> Env:
    env = get_environment(*args, **kwargs)
    return AddAxisWrapper(env)


def get_multiprocess_environment(
    num_environments: int, *args, **kwargs
) -> AsyncVectorEnv:
    envs = AsyncVectorEnv(
        [lambda: get_environment(*args, **kwargs) for _ in range(num_environments)],
        copy=False,
    )
    return envs
