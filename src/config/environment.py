from dataclasses import dataclass


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

    def __post_init__(self) -> None:
        assert self.num_repeat_frames >= 1
        assert self.num_stack_frames >= 1
        assert self.clip_top >= 0
        assert self.clip_bot >= 0
        assert self.clip_left >= 0
        assert self.clip_right >= 0
        assert self.subsampling_factor >= 1
