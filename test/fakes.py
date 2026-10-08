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
        "num_stack_frames": 2,
        "clip_top": 0,
        "clip_bot": 0,
        "clip_left": 0,
        "clip_right": 0,
        "subsampling_factor": 1,
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
        dying_at: int | None = None,
    ) -> None:
        """
        :param dying_at: Step at which Mario starts dying without the episode terminating, like in the full game
        """
        self.rewards = rewards if rewards is not None else [1.0] * 5
        self.world, self.stage = (int(part) for part in level.split("-"))
        self.flag_get = flag_get
        self.truncate = truncate
        self.dying_at = dying_at
        self.observation_space = Box(low=0, high=255, shape=(BASE_HEIGHT, BASE_WIDTH, channels), dtype=np.uint8)
        self.action_space = Discrete(7)
        self.t = 0
        self.num_resets = 0
        self.actions: list[int] = []
        self.powerups: list[int] = []

    def _observation(self) -> np.ndarray:
        columns = np.arange(BASE_WIDTH, dtype=np.uint8)
        return np.broadcast_to(columns[None, :, None], self.observation_space.shape).copy()

    def _info(self, flag_get: bool = False, dying: bool = False) -> dict:
        return {
            "x_pos": 10 * self.t,
            "y_pos": 79,
            "world": self.world,
            "stage": self.stage,
            "area": 1,
            "flag_get": flag_get,
            "is_dying": dying,
            "is_dead": False,
            "reward_components": {"completion": 50.0 if flag_get else 0.0},
        }

    def render(self):
        return self._observation()

    def set_powerup(self, powerup):
        self.powerups.append(int(powerup))
        return self._observation(), self._info()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.num_resets += 1
        return self._observation(), self._info()

    def step(self, action):
        self.actions.append(int(action))
        reward = self.rewards[self.t]
        self.t += 1
        finished = self.t == len(self.rewards)
        terminated, truncated = finished and not self.truncate, finished and self.truncate
        info = self._info(flag_get=terminated and self.flag_get, dying=self.t == self.dying_at)
        return self._observation(), reward, terminated, truncated, info
