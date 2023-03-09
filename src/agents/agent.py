from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy.typing as npt
from omegaconf import DictConfig

from src.summary import Summary


class Agent(ABC):
    def __init__(
        self,
        config: DictConfig,
        in_width: int,
        in_height: int,
        in_stack_frames: int,
        in_channels: int,
        num_workers: int,
        num_actions: int,
        summary: Optional[Summary] = None,
    ) -> None:
        self.config = config
        self.in_width = in_width
        self.in_height = in_height
        self.in_stack_frames = in_stack_frames
        self.in_channels = in_channels
        self.num_workers = num_workers
        self.num_actions = num_actions
        self.summary = summary

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Optional[Path] = None) -> None:
        pass

    def feed_observation(self, state: npt.NDArray) -> None:
        pass

    def update(self) -> None:
        pass

    @abstractmethod
    def next_actions(self, train: bool = True) -> Tuple[List[int], List[float]]:
        pass

    def reset(self) -> None:
        pass

    def give_reward(self, reward: float, done: bool = False) -> None:
        pass

    def set_num_workers(self, num_workers: int) -> None:
        self.num_workers = num_workers
