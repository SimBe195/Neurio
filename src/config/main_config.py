from dataclasses import dataclass
from typing import Any

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
    # _target_: str = ""


# cs = ConfigStore.instance()
# cs.store(name="base_neurio_config", node=NeurioConfig)
# cs.store(name="neurio_config", node=NeurioConfig)
