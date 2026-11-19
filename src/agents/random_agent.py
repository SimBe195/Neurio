import numpy as np

from .agent import Agent


class RandomAgent(Agent):
    """Picks uniformly random actions, as a baseline."""

    def act(self, observations: np.ndarray, greedy: bool = False, untried: np.ndarray | None = None) -> list[int]:
        actions = np.random.randint(self.env_info.num_actions, size=self.num_workers)
        if untried is not None:
            for worker in np.flatnonzero(untried.any(-1)):
                actions[worker] = np.random.choice(np.flatnonzero(untried[worker]))
        return actions.tolist()
