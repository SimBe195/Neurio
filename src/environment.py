from typing import Any, Dict, Tuple
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from gym import Env, ObservationWrapper
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2


class Environment(JoypadSpace):
    def __init__(self) -> None:
        super().__init__(
            gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True),
            SIMPLE_MOVEMENT,
        )


class SubsamplingWrapper(ObservationWrapper):
    def __init__(
        self, env: Env, num_steps: int, new_width: int, new_height: int, **kwargs
    ) -> None:
        super().__init__(env, **kwargs)
        self.num_steps = num_steps
        self.new_width = new_width
        self.new_height = new_height
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(new_width, new_height, num_steps)
        )

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        total_reward = 0
        states = []
        total_terminated = False
        total_truncated = False
        for _ in range(self.num_steps):
            state, reward, terminated, truncated, metrics = super().step(action)
            total_reward += reward
            total_terminated = total_terminated or terminated
            total_truncated = total_truncated or truncated
            states.append(state)
            if total_terminated or total_truncated:
                break
        states = np.concatenate(states, axis=-1)
        states = np.max(states, axis=-1)[..., None]
        return states, total_reward, total_terminated, total_truncated, metrics

    def observation(self, observation: np.array) -> np.array:
        if observation is not None:
            result = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            result = cv2.resize(result, (self.new_width, self.new_height))
            result = result / 255
        else:
            result = np.zeros((self.new_width, self.new_height), dtype=np.float32)
        return result[:, :, None]
