from abc import abstractmethod
from typing import Tuple

import torch
from jaxtyping import Float, Int64

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
    def forward(
        self, x: Float[torch.Tensor, "*batch channel height width"], prev_actions: Int64[torch.Tensor, "*batch"]
    ) -> Tuple[Float[torch.Tensor, "*batch action"], Float[torch.Tensor, "*batch"]]:
        ...
