from typing import List, Tuple

import numpy as np

from src.agents import Agent


class RandomAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)

    def next_actions(self, train: bool = True) -> Tuple[List[int], List[float]]:
        return [
            np.random.randint(0, self.num_actions, dtype=np.int8)
            for _ in range(self.num_workers)
        ], [-np.log(self.num_actions)] * self.num_workers
