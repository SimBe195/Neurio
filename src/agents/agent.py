from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from omegaconf import DictConfig

from src.summary import Summary


class Agent(ABC):
    def __init__(
        self,
        config: DictConfig,
        observation_shape: Tuple[int, ...],
        num_actions: int,
        summary: Summary,
    ) -> None:
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.summary = summary

    @abstractmethod
    def feed_observation(self, state: np.array) -> None:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def next_action(self, train: bool = True) -> int:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def give_reward(self, reward: float, done: bool = False) -> None:
        pass
