from typing import Any, Dict, Optional, Tuple
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from gym import Env, ObservationWrapper
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
import pyglet


class Environment(JoypadSpace):
    def __init__(self) -> None:
        super().__init__(
            gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True),
            SIMPLE_MOVEMENT,
        )


class SubsamplingWrapper(ObservationWrapper):
    def __init__(
        self,
        env: Env,
        num_steps: int,
        render_nn_input: bool = False,
        new_width: Optional[int] = None,
        new_height: Optional[int] = None,
        clip: Tuple[Optional[int], ...] = (None, None, None, None),
        greyscale: bool = True,
        **kwargs
    ) -> None:
        super().__init__(env, **kwargs)
        self.num_steps = num_steps
        self.greyscale = greyscale

        self.render_nn_input = render_nn_input
        self.current_frame = None

        self.new_width = new_width
        self.new_height = new_height

        self.clip_top, self.clip_bot, self.clip_left, self.clip_right = clip
        if not self.new_width:
            self.new_width = 256
            if self.clip_left:
                self.new_width -= self.clip_left
            if self.clip_right:
                self.new_width -= self.clip_right

        if not self.new_height:
            self.new_height = 240
            if self.clip_top:
                self.new_height -= self.clip_top
            if self.clip_bot:
                self.new_height -= self.clip_bot

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(
                self.new_height,
                self.new_width,
                (1 if greyscale else 3),
            ),
        )

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
            state = np.stack([state, state, state], axis=-1)
        return (state * 255).astype(np.int8)

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        total_reward = 0
        total_terminated = False
        total_truncated = False
        max_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        for _ in range(self.num_steps):
            state, reward, terminated, truncated, metrics = super().step(action)
            total_reward += reward
            total_terminated = total_terminated or terminated
            total_truncated = total_truncated or truncated
            max_state = np.maximum(max_state, state)
            if total_terminated or total_truncated:
                break

        if self.render_nn_input:
            self.current_frame = self.to_rgb_frame(state)
        return max_state, total_reward, total_terminated, total_truncated, metrics

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
            result = result / 255
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
