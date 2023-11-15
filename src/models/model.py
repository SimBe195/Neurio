from abc import abstractmethod
from typing import Tuple

import torch

from config.model import ModelConfig
from environment import EnvironmentInfo


class Model(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        env_info: EnvironmentInfo,
    ) -> None:
        super().__init__()
        self.env_info = env_info
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor, prev_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
