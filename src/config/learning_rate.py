from abc import ABC, abstractmethod

import torch
from dataclasses import dataclass


@dataclass
class LRConfig(ABC):
    learning_rate: float

    def __post_init__(self) -> None:
        assert self.learning_rate >= 0

    @abstractmethod
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        ...


@dataclass
class ConstantLRConfig(LRConfig):
    name: str = "constant"

    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1.0, total_iters=0)


@dataclass
class ExponentialDecayLRConfig(LRConfig):
    decay_factor: float
    name: str = "exponential_decay"

    def __post_init__(self) -> None:
        super().__post_init__()
        assert 0 <= self.decay_factor <= 1

    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.decay_factor)


@dataclass
class LinearDecayLRConfig(LRConfig):
    decay_iters: int
    name: str = "linear_decay"

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.decay_iters >= 0

    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=self.decay_iters,
        )
