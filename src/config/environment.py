from dataclasses import dataclass, field


@dataclass
class EnvironmentConfig:
    env_name: str
    complex_movement: bool
    grayscale: bool
    num_repeat_frames: int
    num_stack_frames: int
    clip_top: int
    clip_bot: int
    clip_left: int
    clip_right: int
    subsampling_factor: float
    reward_scale: float
    # Probabilities that an episode on a sampled level other than 1-1 starts small, big or with the fire flower, as
    # if carried over from the previous level. The full game starts small.
    start_powerup_probs: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        assert self.num_repeat_frames >= 1
        assert self.num_stack_frames >= 1
        assert self.clip_top >= 0
        assert self.clip_bot >= 0
        assert self.clip_left >= 0
        assert self.clip_right >= 0
        assert self.subsampling_factor >= 1
        assert self.reward_scale > 0
        assert len(self.start_powerup_probs) == 3 and all(p >= 0 for p in self.start_powerup_probs)
        assert abs(sum(self.start_powerup_probs) - 1) < 1e-6
