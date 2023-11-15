from typing import List, Tuple

import numpy as np

from config import AgentConfig
from environment import EnvironmentInfo
from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, config: AgentConfig, env_info: EnvironmentInfo) -> None:
        super().__init__(config, env_info)

    def next_actions(self, *args, **kwargs) -> Tuple[List[int], List[float]]:
        return [
            np.random.randint(0, self.env_info.num_actions, dtype=np.int8) for _ in range(self.env_info.num_workers)
        ], [-np.log(self.env_info.num_actions)] * self.env_info.num_workers
