from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from omegaconf import DictConfig

from src.summary import Summary


class Agent(ABC):
    def __init__(
        self,
        config: DictConfig,
        num_workers: int,
        num_actions: int,
        summary: Summary,
    ) -> None:
        self.num_workers = num_workers
        self.num_actions = num_actions
        self.summary = summary

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def feed_observation(self, state: np.array) -> None:
        pass

    def update(self) -> None:
        pass

    @abstractmethod
    def next_actions(self, train: bool = True) -> List[int]:
        pass

    def reset(self) -> None:
        pass

    def give_reward(self, reward: float, done: bool = False) -> None:
        pass

    def set_num_workers(self, num_workers: int) -> None:
        self.num_workers = num_workers
