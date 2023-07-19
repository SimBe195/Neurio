from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from hydra.core.config_store import ConfigStore


@dataclass
class LRConfig(ABC):
    learning_rate: float

    def __post_init__(self) -> None:
        assert self.learning_rate >= 0

    @abstractmethod
    def create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        ...


@dataclass
class ConstantLRConfig(LRConfig):
    def create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer=optimizer, factor=1.0, total_iters=0
        )


@dataclass
class ExponentialDecayLRConfig(LRConfig):
    decay_factor: float

    def __post_init__(self) -> None:
        super().__post_init__()
        assert 0 <= self.decay_factor <= 1

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=self.decay_factor
        )


@dataclass
class LinearDecayLRConfig(LRConfig):
    decay_iters: int

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.decay_iters >= 0

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=self.decay_iters,
        )


# cs = ConfigStore.instance()
# cs.store(group="agent/learning_rate", name="base_constant", node=ConstantLRConfig)
# cs.store(
#     group="agent/learning_rate",
#     name="base_exponential_decay",
#     node=ExponentialDecayLRConfig,
# )
# cs.store(
#     group="agent/learning_rate", name="base_linear_decay", node=LinearDecayLRConfig
# )
