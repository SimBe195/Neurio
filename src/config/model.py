from dataclasses import dataclass


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
