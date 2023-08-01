from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    pass


@dataclass
class ActorCriticConfig(ModelConfig):
    action_embedding_size: int
    num_layers: int
    num_filters: int
    kernel_size: int
    stride: int
    fc_size: int

    def __post_init__(self) -> None:
        assert self.action_embedding_size > 0
        assert self.num_layers > 0
        assert self.num_filters > 0
        assert self.kernel_size > 0
        assert self.stride > 0
        assert self.fc_size > 0


# cs = ConfigStore.instance()
# cs.store(group="agent/model", name="base_actor_critic", node=ActorCriticConfig)