from typing import List

import numpy as np

from src.agents import Agent


class RandomAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)

    def next_actions(self, train: bool = True) -> List[int]:
        return [
            np.random.randint(0, self.num_actions, dtype=np.int8)
            for _ in range(self.num_workers)
        ]
