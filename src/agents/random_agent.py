import numpy as np

from src.agents import Agent


class RandomAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)

    def next_action(self, train: bool = True) -> int:
        return np.random.randint(0, self.num_actions, dtype=np.int8)
