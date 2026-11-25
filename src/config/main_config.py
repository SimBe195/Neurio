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
    exploration: ExplorationConfig
    num_workers: int
    num_iters: int
    steps_per_iter: int
    save_frequency: int
    # Iterations between saves of the full state of training to resume from, which include the replay, see resume.py
    resume_frequency: int
    tracking_uri: str
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
        assert self.resume_frequency > 0
        if self.test_iter is not None:
            assert self.test_iter >= 0
