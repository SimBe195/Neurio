from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from config.agent import AgentConfig
from environment import EnvironmentInfo


@dataclass
class StepOutcome:
    """What the last actions of all workers led to, per worker."""

    rewards: np.ndarray
    # The episode ended in the game, e.g. by dying or reaching the flag
    terminated: np.ndarray
    # The episode was cut off, e.g. by a time limit
    truncated: np.ndarray
    # Level of each worker; for finished episodes, the level they ended on
    levels: list[str]
    # Last observation of finished episodes, None for the other workers
    final_states: np.ndarray | None = None
    # Exploration bonus on top of the game's reward
    bonus: np.ndarray | None = None

    @property
    def dones(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)


class Agent(ABC):
    """Acts for all workers at once. The hooks without abstract decorator are optional."""

    def __init__(self, config: AgentConfig, env_info: EnvironmentInfo) -> None:
        self.config = config
        self.env_info = env_info

    @property
    def num_workers(self) -> int:
        return self.env_info.num_workers

    def describe(self) -> str | None:
        """Description of the agent's model, logged with the training run."""
        return None

    def save(self, save_iter: int) -> None:
        """Save a checkpoint to the active mlflow run."""

    def load(self, run_id: str, load_iter: int) -> None:
        """Load a checkpoint of the given mlflow run."""

    def state_dict(self) -> dict:
        """Everything training changed, to resume it."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def feed_observation(self, state: np.ndarray) -> None:
        pass

    @abstractmethod
    def next_actions(self) -> tuple[list[int], list[float]]:
        """Indices and log probabilities of the next actions for each worker."""

    def give_reward(self, outcome: StepOutcome) -> None:
        pass

    def update(self, next_states: np.ndarray, episode_reward: float | None = None) -> dict[str, float]:
        """
        Update the agent from its buffered experience.

        :param next_states: Observations reached after the last buffered step, used for bootstrapping
        :param episode_reward: Mean reward of the episodes finished since the last update, None if there were none
        :return: Training metrics of the update
        """
        return {}
