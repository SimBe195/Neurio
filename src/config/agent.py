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
    batch_size: int
    clip_param: float
    clip_value: bool
    critic_loss_weight: float
    initial_entropy_loss_weight: float
    final_entropy_loss_weight: float
    entropy_decay_updates: int
    grad_clip_norm: float
    normalize_advantages: bool = True
    adam_eps: float = 1e-5
    # Self-imitation learning: after each update, imitate the agent's own sil_episodes_per_level best episodes per
    # level and start power-up where their return beats the current value estimate. A loss weight of 0 disables it.
    sil_loss_weight: float = 1.0
    sil_value_loss_weight: float = 0.01
    sil_episodes_per_level: int = 16
    sil_updates: int = 8
    sil_batch_size: int = 256

    def __post_init__(self) -> None:
        assert 0 <= self.gamma <= 1
        assert 0 <= self.tau <= 1
        assert self.exp_buffer_size >= 1
        assert self.epochs_per_update >= 1
        assert self.batch_size >= 1
        assert 0 <= self.clip_param <= 1
        assert self.initial_entropy_loss_weight >= 0
        assert self.final_entropy_loss_weight >= 0
        assert self.entropy_decay_updates >= 1
        assert self.adam_eps > 0
        assert self.sil_loss_weight >= 0
        assert self.sil_value_loss_weight >= 0
        assert self.sil_episodes_per_level >= 1
        assert self.sil_updates >= 1
        assert self.sil_batch_size >= 1
