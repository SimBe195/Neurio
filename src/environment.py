from typing import Any, Dict, Tuple

import cv2
import gym_super_mario_bros
import numpy as np
import pyglet
from gym.core import Env, ObservationWrapper
from gym.spaces import Box
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from omegaconf import DictConfig

from src.reward_func import CustomRewardWrapper


class BaseEnvironment(JoypadSpace):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(
            gym_super_mario_bros.make(config.env_name, new_step_api=True),
            COMPLEX_MOVEMENT if config.complex_movement else SIMPLE_MOVEMENT,
        )


class SubsamplingWrapper(ObservationWrapper):
    def __init__(self, config: DictConfig, env: Env, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.num_frames = config.num_frames
        self.greyscale = config.greyscale

        self.render_nn_input = config.render_nn_input

        self.reduce_factor = config.reduce_factor

        self.clip_top = config.clip_top
        self.clip_bot = config.clip_bot
        self.clip_left = config.clip_left
        self.clip_right = config.clip_right

        self.new_width = (256 - self.clip_left - self.clip_right) // self.reduce_factor
        self.new_height = (240 - self.clip_top - self.clip_bot) // self.reduce_factor

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(
                self.new_height,
                self.new_width,
                (1 if self.greyscale else 3),
            ),
        )
        self.current_frame = np.zeros(self.observation_space.shape, dtype=np.float32)

        if self.render_nn_input:
            self._window = pyglet.window.Window(
                caption="Neural Network input image",
                height=self.new_height,
                width=self.new_width,
                vsync=False,
                resizable=True,
            )

    def to_rgb_frame(self, state: np.array) -> np.array:
        if self.greyscale:
            # duplicate value to R, G and B channel
            state = np.stack([state, state, state], axis=-1)
        return state.astype(np.int8)

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        total_reward = 0
        for _ in range(self.num_frames):
            state, reward, terminated, truncated, metrics = super().step(action)
            total_reward += reward
            if terminated or truncated:
                break

        if self.render_nn_input:
            self.current_frame = self.to_rgb_frame(state)
        return state, total_reward, terminated, truncated, metrics

    def reset(self) -> np.array:
        state = super().reset()
        if self.render_nn_input:
            self.current_frame = self.to_rgb_frame(state)
        return state

    def observation(self, observation: np.array) -> np.array:
        if observation is not None:
            result = observation[
                self.clip_top : -self.clip_bot if self.clip_bot else None,
                self.clip_left : -self.clip_right if self.clip_right else None,
                ...,
            ]
            result = cv2.resize(result, (self.new_width, self.new_height))
            if self.greyscale:
                result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)[..., None]
            result = result.astype(np.float32)
        else:
            result = np.zeros(self.observation_space.shape, dtype=np.float32)
        return result

    def render(self, *args, **kwargs) -> Any:
        if self.render_nn_input:
            self._window.clear()
            self._window.switch_to()
            self._window.dispatch_events()
            image = pyglet.image.ImageData(
                self.current_frame.shape[1],
                self.current_frame.shape[0],
                "RGB",
                self.current_frame.tobytes(),
                pitch=self.current_frame.shape[1] * -3,
            )
            # send the image to the window
            image.blit(0, 0, width=self._window.width, height=self._window.height)
            self._window.flip()

        return super().render(*args, **kwargs)


def get_environment(config: DictConfig) -> BaseEnvironment:
    env = BaseEnvironment(config)
    env = CustomRewardWrapper(config.reward, env)
    env = SubsamplingWrapper(config, env)
    return env
