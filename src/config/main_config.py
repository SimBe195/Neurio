from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore

from .agent import AgentConfig
from .environment import EnvironmentConfig


@dataclass
class NeurioConfig:
    environment: EnvironmentConfig
    agent: AgentConfig
    level: str
    num_workers: int
    num_iters: int
    steps_per_iter: int
    save_frequency: int
    render: bool
    test_run_id: Optional[str] = None
    test_iter: Optional[int] = None
    # _target_: str = ""

    def __post_init__(self) -> None:
        assert self.num_workers > 0
        assert self.num_iters > 0
        assert self.steps_per_iter > 0
        assert self.save_frequency > 0
        if self.test_iter is not None:
            assert self.test_iter >= 0


# cs = ConfigStore.instance()
# cs.store(name="base_neurio_config", node=NeurioConfig)
# cs.store(name="neurio_config", node=NeurioConfig)
