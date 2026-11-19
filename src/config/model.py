from dataclasses import dataclass


@dataclass
class NatureActorCriticConfig:
    """Actor-critic with the DQN encoder ("Nature CNN", Mnih et al., 2015), see agents.ppo.nature_encoder."""

    action_embedding_size: int
    num_filters: list[int]
    kernel_sizes: list[int]
    strides: list[int]
    fc_size: int

    def __post_init__(self) -> None:
        assert self.action_embedding_size > 0
        assert all(f > 0 for f in self.num_filters)
        assert all(k > 0 for k in self.kernel_sizes)
        assert all(s > 0 for s in self.strides)
        assert len(self.num_filters) == len(self.kernel_sizes) == len(self.strides)
        assert self.fc_size > 0


@dataclass
class ImpalaActorCriticConfig:
    """Actor-critic with the residual encoder of IMPALA, see agents.ppo.impala_encoder."""

    action_embedding_size: int
    # Output channels of each stage, each of which halves the resolution
    channels: list[int]
    blocks_per_stage: int
    fc_size: int

    def __post_init__(self) -> None:
        assert self.action_embedding_size > 0
        assert self.channels and all(c > 0 for c in self.channels)
        assert self.blocks_per_stage >= 0
        assert self.fc_size > 0
