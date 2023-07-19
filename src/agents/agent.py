from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy.typing as npt

from src.config.agent import AgentConfig
from src.environment import EnvironmentInfo


class Agent(ABC):
    def __init__(
        self,
        config: AgentConfig,
        env_info: EnvironmentInfo,
    ) -> None:
        self.config = config
        self.env_info = env_info

    def save(self, iter: int) -> None:
        pass

    def load(self, iter: int) -> None:
        pass

    def feed_observation(self, state: npt.NDArray) -> None:
        pass

    def update(self) -> None:
        pass

    @property
    def num_workers(self) -> int:
        return self.env_info.num_workers

    @abstractmethod
    def next_actions(self, train: bool = True) -> Tuple[List[int], List[float]]:
        """
        Gets indices and log probabilities of the next actions for each worker.
        """
        pass

    def reset(self) -> None:
        pass

    def give_reward(self, reward: float, done: bool = False) -> None:
        pass
