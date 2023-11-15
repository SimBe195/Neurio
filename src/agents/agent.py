from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from config.agent import AgentConfig
from environment import EnvironmentInfo


class Agent(ABC):
    def __init__(
        self,
        config: AgentConfig,
        env_info: EnvironmentInfo,
    ) -> None:
        self.config = config
        self.env_info = env_info

    def save(self, save_iter: int) -> None:
        pass

    def load(self, load_iter: int) -> None:
        pass

    def feed_observation(self, state: np.ndarray) -> None:
        pass

    def update(self) -> None:
        pass

    @property
    def num_workers(self) -> int:
        return self.env_info.num_workers

    @abstractmethod
    def next_actions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gets indices and log probabilities of the next actions for each worker.
        """
        pass

    def reset(self) -> None:
        pass

    def give_reward(self, reward: np.ndarray, done: np.ndarray) -> None:
        pass
