import numpy as np

from config import AgentConfig
from environment import EnvironmentInfo

from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, config: AgentConfig, env_info: EnvironmentInfo) -> None:
        super().__init__(config, env_info)

    def next_actions(self, *args, **kwargs) -> tuple[list[int], list[float]]:
        return [int(np.random.randint(0, self.env_info.num_actions)) for _ in range(self.env_info.num_workers)], [
            -np.log(self.env_info.num_actions)
        ] * self.env_info.num_workers
