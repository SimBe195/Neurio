from dataclasses import dataclass

from .learning_rate import LRConfig
from .model import ModelConfig
from .sampling_strategy import SamplingStrategy


@dataclass
class AgentConfig:
    pass


@dataclass
class RandomAgentConfig(AgentConfig):
    pass


@dataclass
class PPOAgentConfig(AgentConfig):
    model: ModelConfig
    learning_rate: LRConfig
    sampling_strategy: SamplingStrategy
    gamma: float
    tau: float
    exp_buffer_size: int
    epochs_per_update: int
    total_updates: int
    batch_size: int
    clip_param: float
    clip_value: bool
    critic_loss_weight: float
    max_entropy_loss_weight: float
    grad_clip_norm: float

    def __post_init__(self) -> None:
        assert 0 <= self.gamma <= 1
        assert 0 <= self.tau <= 1
        assert self.exp_buffer_size >= 1
        assert self.epochs_per_update >= 1
        assert self.total_updates >= 1
        assert self.batch_size >= 1
        assert 0 <= self.clip_param <= 1
