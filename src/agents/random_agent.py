from typing import List, Tuple

import numpy as np

from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)

    def next_actions(self, *args, **kwargs) -> Tuple[List[int], List[float]]:
        return [
            np.random.randint(0, self.env_info.num_actions, dtype=np.int8)
            for _ in range(self.env_info.num_workers)
        ], [-np.log(self.env_info.num_actions)] * self.env_info.num_workers
