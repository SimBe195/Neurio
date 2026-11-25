from dataclasses import dataclass
from typing import ClassVar

from .model import ImpalaActorCriticConfig


@dataclass
class AgentConfig:
    @property
    def stack_frames(self) -> int:
        """
        Number of most recent frames in each observation, so that a network without memory can see motion. Agents
        that remember earlier frames themselves need only the current one.
        """
        return 1


@dataclass
class PPOAgentConfig(AgentConfig):
    model: ImpalaActorCriticConfig
    learning_rate: float
    num_stack_frames: int
    gamma: float
    # Generalized advantage estimation's lambda
    lam: float
    epochs_per_update: int
    batch_size: int
    clip_param: float
    critic_loss_weight: float
    initial_entropy_loss_weight: float
    final_entropy_loss_weight: float
    entropy_decay_updates: int
    grad_clip_norm: float
    adam_eps: float
    # Share of the uniform distribution mixed into the policy, so that no action becomes impossible, as in DreamerV3
    unimix: float
    # Self-imitation learning: after each update, imitate the agent's own sil_episodes_per_level best episodes per
    # level where their return beats the current value estimate
    sil_loss_weight: float
    sil_value_loss_weight: float
    sil_episodes_per_level: int
    sil_updates: int
    sil_batch_size: int

    @property
    def stack_frames(self) -> int:
        return self.num_stack_frames

    def __post_init__(self) -> None:
        assert 0 <= self.gamma <= 1
        assert 0 <= self.lam <= 1
        assert self.learning_rate > 0
        assert 0 <= self.unimix < 1
        assert self.num_stack_frames >= 1
        assert self.epochs_per_update >= 1
        assert self.batch_size >= 1
        assert 0 <= self.clip_param <= 1
        assert self.initial_entropy_loss_weight >= 0
        assert self.final_entropy_loss_weight >= 0
        assert self.entropy_decay_updates >= 1
        assert self.adam_eps > 0
        assert self.sil_loss_weight > 0
        assert self.sil_value_loss_weight >= 0
        assert self.sil_episodes_per_level >= 1
        assert self.sil_updates >= 1
        assert self.sil_batch_size >= 1


@dataclass
class DreamerAgentConfig(AgentConfig):
    """DreamerV3 (Hafner et al., 2023), see agents.dreamer and configs/agent/dreamer.yaml."""

    # The losses of agents.dreamer.DreamerAgent, each weighted by loss_scales
    LOSSES: ClassVar[set[str]] = {
        "reconstruction",
        "reward",
        "continuation",
        "dynamics",
        "representation",
        "policy",
        "value",
        "replay_value",
        "self_imitation",
        "disagreement",
        "exploration_value",
        "cmpo",
    }

    deter: int
    blocks: int
    hidden: int
    stoch: int
    classes: int
    depth: int
    mults: list[int]
    kernel: int
    decoder_blocks: int
    units: int
    actor_critic_layers: int
    head_layers: int
    bins: int
    unimix: float
    free_nats: float
    loss_scales: dict[str, float]
    batch_size: int
    batch_length: int
    train_ratio: float
    replay_size: int
    learning_rate: float
    agc: float
    warmup: int
    imagination_length: int
    horizon: int
    lam: float
    entropy_weight: float
    # Label smoothing of the target of Muesli's CMPO regularizer (loss_scales.cmpo), as in supervised learning: unlike
    # the entropy bonus, it pulls up actions that the actor gives almost no probability, by smoothing / actions
    cmpo_label_smoothing: float
    slow_regularization: float
    slow_critic_rate: float
    return_percentiles: list[float]
    return_norm_rate: float
    # Share of the replayed sequences, i.e. the batch without self-imitation, that are sampled around a step picked by
    # how rarely the agent has been there, see StepOutcome.rarity
    rare_fraction: float
    # Self-imitation: share of each batch from the agent's own best episodes, of which it keeps this many per level
    sil_fraction: float
    sil_episodes_per_level: int
    # Plan2Explore: ensemble whose disagreement about the next state is rewarded in imagination, weighted against the
    # game's reward after both are normalized
    exploration_heads: int
    exploration_weight: float

    @property
    def discount(self) -> float:
        """Discount of future rewards per step."""
        return 1 - 1 / self.horizon

    def __post_init__(self) -> None:
        assert self.deter % self.blocks == 0
        assert self.stoch > 0 and self.classes > 0
        assert self.bins % 2 == 1
        assert self.batch_size >= 1 and self.batch_length >= 2
        assert self.train_ratio > 0
        assert self.replay_size >= self.batch_length
        assert self.imagination_length >= 1
        assert 0 < self.unimix < 1
        assert set(self.loss_scales) == self.LOSSES, f"loss_scales must have the scales of {sorted(self.LOSSES)}"
        assert 0 <= self.cmpo_label_smoothing < 1
        assert len(self.return_percentiles) == 2
        assert 0 < self.sil_fraction < 1 and self.sil_episodes_per_level >= 1
        assert 0 <= self.rare_fraction <= 1
        assert self.exploration_heads >= 2 and self.exploration_weight > 0
