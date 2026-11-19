import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete

from config.environment import EnvironmentConfig
from environment import BASE_HEIGHT, BASE_WIDTH


def make_env_config(**overrides) -> EnvironmentConfig:
    values = {
        "env_name": "SuperMarioBros-v0",
        "complex_movement": False,
        "grayscale": True,
        "num_repeat_frames": 1,
        "clip_top": 0,
        "clip_bot": 0,
        "clip_left": 0,
        "clip_right": 0,
        "height": BASE_HEIGHT,
        "width": BASE_WIDTH,
        "reward_scale": 0.1,
    }
    values.update(overrides)
    return EnvironmentConfig(**values)


class FakeMarioEnv(gymnasium.Env):
    """
    Stand-in for the NES environment. Observations encode the column index, so that orientation of the
    image can be checked after preprocessing. Episodes follow a scripted list of rewards per step.
    """

    def __init__(
        self,
        rewards: list[float] | None = None,
        level: str = "1-1",
        flag_get: bool = False,
        channels: int = 3,
        truncate: bool = False,
    ) -> None:
        self.rewards = rewards if rewards is not None else [1.0] * 5
        self.world, self.stage = (int(part) for part in level.split("-"))
        self.flag_get = flag_get
        self.truncate = truncate
        self.observation_space = Box(low=0, high=255, shape=(BASE_HEIGHT, BASE_WIDTH, channels), dtype=np.uint8)
        self.action_space = Discrete(7)
        self.t = 0
        self.num_resets = 0
        self.actions: list[int] = []
        # Power-ups given at the start of episodes, see environment.BaseEnvironment.set_powerup
        self.powerups: list[int] = []
        self.powerup = 0

    def _observation(self) -> np.ndarray:
        columns = np.arange(BASE_WIDTH, dtype=np.uint8)
        return np.broadcast_to(columns[None, :, None], self.observation_space.shape).copy()

    def _info(self, flag_get: bool = False) -> dict:
        return {
            "x_pos": 10 * self.t,
            "y_pos": 79,
            "world": self.world,
            "stage": self.stage,
            "area": 1,
            "flag_get": flag_get,
            "life": 2,
            "on_ground": True,
            "powerup_level": self.powerup,
            "reward_components": {"completion": 50.0 if flag_get else 0.0},
        }

    def render(self):
        return self._observation()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.num_resets += 1
        self.powerup = 0
        return self._observation(), self._info()

    def set_powerup(self, powerup: int):
        self.powerups.append(powerup)
        self.powerup = powerup
        return self._observation(), self._info()

    def step(self, action):
        self.actions.append(int(action))
        reward = self.rewards[self.t]
        self.t += 1
        finished = self.t == len(self.rewards)
        terminated, truncated = finished and not self.truncate, finished and self.truncate
        info = self._info(flag_get=terminated and self.flag_get)
        return self._observation(), reward, terminated, truncated, info


class WarpingMarioEnv(FakeMarioEnv):
    """Moves 10 pixels per step, but goes back to x = 0 of a new area, or of level 1-2, at step 2."""

    def __init__(self, new_level: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.new_level = new_level

    def _info(self, **kwargs) -> dict:
        info = super()._info(**kwargs)
        if self.t < 2:
            return info
        return {**info, "x_pos": 10 * (self.t - 2), "stage": 1 + self.new_level, "area": 1 + (not self.new_level)}
