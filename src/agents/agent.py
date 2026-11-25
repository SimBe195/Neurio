import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import mlflow
import numpy as np
import torch

from config.agent import AgentConfig
from environment import EnvironmentInfo, Powerup

log = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "checkpoint-"


class Play(Enum):
    """How an agent acts."""

    # Samples its actions and learns from them
    TRAIN = "train"
    # Samples its actions without learning, to evaluate the policy as it trains
    SAMPLE = "sample"
    # Takes its most likely actions without learning
    GREEDY = "greedy"


def checkpoint_name(iteration: int) -> str:
    return f"{CHECKPOINT_PREFIX}{iteration:04d}"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        log.info(f"Using GPU {torch.cuda.get_device_name()}")
        # The input size is fixed, so the fastest convolution algorithms can be chosen once
        torch.backends.cudnn.benchmark = True
        # TensorFloat-32 for matrix multiplications in full precision
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    log.info("Using CPU")
    return torch.device("cpu")


@dataclass
class StepOutcome:
    """What the last actions of all workers led to, per worker."""

    # Observations the workers continue from. For finished episodes, the first observation of the next one.
    observations: np.ndarray
    rewards: np.ndarray
    # The episode ended in the game, e.g. by dying or reaching the flag
    terminated: np.ndarray
    # The episode was cut off, e.g. by a time limit
    truncated: np.ndarray
    # Level of each worker; for finished episodes, the level they ended on
    levels: list[str]
    # The game's reward, by which episodes are compared, e.g. to keep the best ones. The agent learns from `rewards`.
    game_rewards: np.ndarray
    # Mario's power-up at the start of each worker's episode, see environment.Powerup: episodes are only compared with
    # those of the same level and start
    start_powerups: np.ndarray
    # Last observation of finished episodes, None while no episode finished
    final_observations: np.ndarray | None = None
    # How rarely the workers have been where they are: 1 / visits of the place, see exploration.NoveltyBonus. None
    # without the bonus, e.g. when evaluating.
    rarity: np.ndarray | None = None
    # For workers whose episode finished, how rare the place where the next one starts is
    first_rarity: np.ndarray | None = None
    # The novelty bonus included in `rewards`, None without it
    bonuses: np.ndarray | None = None

    @property
    def dones(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)

    @property
    def task_rewards(self) -> np.ndarray:
        """The rewards without the novelty bonus, which wears off: what the agent's best episodes are imitated for."""
        return self.rewards if self.bonuses is None else self.rewards - self.bonuses

    @property
    def episode_groups(self) -> list[str]:
        """Per worker, the episodes its episode is compared with: those of the same level and start, e.g. "1-2 fire"."""
        return [
            f"{level} {Powerup(int(powerup)).name.lower()}"
            for level, powerup in zip(self.levels, self.start_powerups, strict=True)
        ]


class Agent(ABC):
    """
    Acts for all workers at once. Each step, the game loop asks for actions with `act` and reports what they led to
    with `observe`. While the workers play the actions, `train_while_playing` may learn from the experience so far, and
    after a number of steps, `update` learns from the rest.
    """

    def __init__(self, config: AgentConfig, env_info: EnvironmentInfo) -> None:
        self.config = config
        self.env_info = env_info

    @property
    def num_workers(self) -> int:
        return self.env_info.num_workers

    @abstractmethod
    def act(self, observations: np.ndarray, play: Play = Play.TRAIN, untried: np.ndarray | None = None) -> list[int]:
        """
        :param observations: The current observation of each worker
        :param untried: Per worker and action, whether the action has never been taken where the worker is, see
            exploration.UntriedActions. A worker with any takes one of them, in the order of its policy.
        :return: The next action of each worker
        """

    def observe(self, outcome: StepOutcome) -> None:
        """What the actions returned by the last `act` led to."""

    def reset(self) -> None:
        """Start new episodes on all workers, whose environments were reset."""

    def train_while_playing(self) -> None:
        """Learn while the workers play the actions of the last `act`, which leaves the GPU idle otherwise."""

    def update(self) -> dict[str, float]:
        """Learn from the experience since the last update. :return: Training metrics, also of `train_while_playing`"""
        return {}

    def describe(self) -> str | None:
        """Description of the agent's networks, logged with the training run."""
        return None

    @property
    def networks(self) -> dict[str, torch.nn.Module]:
        """The trained networks, which checkpoints store."""
        return {}

    def state_dict(self) -> dict:
        """Everything training changed, to resume it."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def save(self, iteration: int) -> None:
        """Save a checkpoint of the networks to the active mlflow run."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "networks.pt"
            torch.save({name: network.state_dict() for name, network in self.networks.items()}, path)
            mlflow.log_artifact(str(path), checkpoint_name(iteration))

    def load(self, run_id: str, iteration: int) -> None:
        """Load the networks of a checkpoint of the given mlflow run."""
        with tempfile.TemporaryDirectory() as directory:
            path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"{checkpoint_name(iteration)}/networks.pt", dst_path=directory
            )
            states = torch.load(path, weights_only=True, map_location="cpu")
        for name, network in self.networks.items():
            network.load_state_dict(states[name])
