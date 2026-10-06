from dataclasses import dataclass, field

from .agent import AgentConfig
from .environment import EnvironmentConfig
from .exploration import ExplorationConfig
from .levels import LevelConfig


@dataclass
class NeurioConfig:
    environment: EnvironmentConfig
    agent: AgentConfig
    levels: LevelConfig
    num_workers: int
    num_iters: int
    steps_per_iter: int
    save_frequency: int
    render: bool
    tracking_uri: str
    objective_window: int
    # "recent_mean_reward": mean reward of the last objective_window episodes.
    # "levels_cleared_score": the level scheduler's expected number of levels cleared, for curricula.
    # "mean_levels_cleared_score": its mean over all iterations so far, which rewards learning fast and steadily.
    objective: str = "recent_mean_reward"
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    # Stored as mlflow tags of the run, e.g. to group experiments. Set with `+tags.<name>=<value>`.
    tags: dict[str, str] = field(default_factory=dict)
    # Whether to continue an interrupted run with the same config from its last checkpoint, None to ask
    resume: bool | None = None
    test_run_id: str | None = None
    test_iter: int | None = None

    def __post_init__(self) -> None:
        assert self.num_workers > 0
        assert self.num_iters > 0
        assert self.steps_per_iter > 0
        assert self.save_frequency > 0
        assert self.objective_window > 0
        assert self.objective in ("recent_mean_reward", "levels_cleared_score", "mean_levels_cleared_score")
        if self.test_iter is not None:
            assert self.test_iter >= 0
