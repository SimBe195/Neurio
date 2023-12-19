from dataclasses import dataclass


@dataclass
class ModelConfig:
    pass


@dataclass
class ActorCriticConfig(ModelConfig):
    action_embedding_size: int
    num_filters: list[int]
    kernel_sizes: list[int]
    fc_size: int

    def __post_init__(self) -> None:
        assert self.action_embedding_size > 0
        assert all(f > 0 for f in self.num_filters)
        assert all(k > 0 for k in self.kernel_sizes)
        assert len(self.num_filters) == len(self.kernel_sizes)
        assert self.fc_size > 0
