from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    env_name: str
    complex_movement: bool
    grayscale: bool
    num_repeat_frames: int
    # Borders cut off the frame, e.g. the status bar, before it is resized to height x width
    clip_top: int
    clip_bot: int
    clip_left: int
    clip_right: int
    height: int
    width: int
    # Scale of all rewards, given in the game's units: the game pays about 1 per pixel of progress
    reward_scale: float
    # What the agent learns from, see environment.ProgressShaping: reaching the flag, dying, and as guidance towards
    # the flag the progress to the right, per pixel. The game's own are 50, 25 and 1.
    flag_reward: float = 50.0
    death_penalty: float = 25.0
    progress_reward: float = 1.0
    # Whether to paint everything black that Mario cannot touch, see environment.remove_background
    remove_background: bool = False
    # Probability that the game keeps the previous frame's action instead of the chosen one, at each frame
    sticky_action_probability: float = 0.0
    # Steps, i.e. repeated actions, after which an episode without a new furthest position is truncated. None waits
    # for the game's time to run out.
    max_steps_without_progress: int | None = None

    def __post_init__(self) -> None:
        assert self.num_repeat_frames >= 1
        assert 0 <= self.sticky_action_probability < 1
        assert self.clip_top >= 0
        assert self.clip_bot >= 0
        assert self.clip_left >= 0
        assert self.clip_right >= 0
        assert self.height > 0 and self.width > 0
        assert self.reward_scale > 0
        assert self.flag_reward >= 0
        assert self.death_penalty >= 0
        assert self.progress_reward >= 0
        assert self.max_steps_without_progress is None or self.max_steps_without_progress >= 1
