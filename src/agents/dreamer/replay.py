from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch


@dataclass
class Latent:
    """The model state of steps, as stored: the deterministic part in half precision, the stochastic one as classes."""

    deter: npt.NDArray[np.float16]
    stoch: npt.NDArray[np.uint8]


class Replay:
    """
    The most recent steps of each worker, from which training samples sequences of consecutive steps. Each step holds
    the frame, the reward received on reaching it, whether it starts or ends an episode, the action taken in it and
    the model state the agent had in it.

    Sequences can span the end of one episode and the start of the next, which `is_first` marks. As in DreamerV3,
    each new sequence of a worker is sampled once as soon as it is complete ("online"), all others uniformly, and it
    starts from the state the agent had in the step before, so that the model learns its first steps in context.

    Rare experience can be sampled more often: around a step picked by the rarity stored with it, e.g. 1 / visits of
    Mario's place, which the uniform sampling would otherwise only rarely replay.
    """

    def __init__(
        self,
        capacity: int,
        num_workers: int,
        frame_shape: tuple[int, ...],
        length: int,
        deter: int,
        stoch: int,
        seed: int | None = None,
    ) -> None:
        """
        :param deter: Size of the deterministic model state
        :param stoch: Number of stochastic variables of the model state
        """
        self.size = capacity // num_workers
        assert self.size >= length, "The replay must hold at least one sequence per worker."
        self.length = length
        self.num_workers = num_workers
        self.frames = np.zeros((num_workers, self.size, *frame_shape), np.uint8)
        self.actions = np.zeros((num_workers, self.size), np.int64)
        self.rewards = np.zeros((num_workers, self.size), np.float32)
        self.is_first = np.zeros((num_workers, self.size), bool)
        self.is_last = np.zeros((num_workers, self.size), bool)
        self.is_terminal = np.zeros((num_workers, self.size), bool)
        self.rarity = np.ones((num_workers, self.size), np.float32)
        self.deter = np.zeros((num_workers, self.size, deter), np.float16)
        self.stoch = np.zeros((num_workers, self.size, stoch), np.uint8)
        # Number of steps added per worker, the next one goes to index added % size
        self.added = np.zeros(num_workers, np.int64)
        self.online: deque[tuple[int, int]] = deque()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(np.minimum(self.added, self.size).sum())

    def add(
        self,
        worker: int,
        frame: npt.NDArray[np.uint8],
        action: int,
        reward: float,
        is_first: bool,
        latent: Latent | None,
        is_last: bool = False,
        is_terminal: bool = False,
        rarity: float = 1.0,
    ) -> None:
        """
        :param latent: The model state in the step, None for the final frame of an episode, in which none acts
        :param rarity: How rare the step is, by which `sample` picks rare sequences
        """
        index = self.added[worker] % self.size
        self.frames[worker, index] = frame
        self.actions[worker, index] = action
        self.rewards[worker, index] = reward
        self.is_first[worker, index] = is_first
        self.is_last[worker, index] = is_last
        self.is_terminal[worker, index] = is_terminal
        self.rarity[worker, index] = rarity
        self.deter[worker, index] = 0 if latent is None else latent.deter
        self.stoch[worker, index] = 0 if latent is None else latent.stoch
        self.added[worker] += 1
        if self.added[worker] % self.length == 0:
            self.online.append((worker, int(self.added[worker] - self.length)))

    # The arrays of the steps, per worker
    STEP_ARRAYS = ("frames", "actions", "rewards", "is_first", "is_last", "is_terminal", "rarity", "deter", "stoch")

    def state_dict(self) -> dict:
        """The stored steps, as tensors that share the arrays' memory."""
        arrays = {name: torch.from_numpy(getattr(self, name)) for name in (*self.STEP_ARRAYS, "added")}
        return {**arrays, "online": list(self.online)}

    def load_state_dict(self, state: dict) -> None:
        for name in (*self.STEP_ARRAYS, "added"):
            np.copyto(getattr(self, name), state[name].numpy())
        self.online = deque(tuple(sequence) for sequence in state["online"])
        # The episodes in progress were cut off, which the last step of each worker marks like a truncation
        workers = np.flatnonzero(self.added)
        self.is_last[workers, (self.added[workers] - 1) % self.size] = True

    def can_sample(self) -> bool:
        return bool((self.added >= self.length).any())

    def sample(self, batch_size: int, online: bool = True, num_rare: int = 0) -> dict[str, np.ndarray]:
        """
        :param online: Whether new sequences come first, as for training
        :param num_rare: Number of sequences, the last ones, around a step picked with probability by its rarity
        :return: Sequences of shape (batch, length, ...) for "frames", "actions", "rewards", "is_first", "is_last" and
            "is_terminal", and of shape (batch, ...) the step before each: its "start_deter", "start_stoch" and
            "start_action"
        """
        starts = [self._sample_start(online) for _ in range(batch_size - num_rare)] + self._sample_rare_starts(num_rare)
        workers = np.array([worker for worker, _ in starts])
        first = np.array([start for _, start in starts])
        indices = (first[:, None] + np.arange(self.length)) % self.size
        batch = {
            name: getattr(self, name)[workers[:, None], indices]
            for name in ("frames", "actions", "rewards", "is_first", "is_last", "is_terminal")
        }
        before = (first - 1) % self.size
        batch.update(
            start_deter=self.deter[workers, before],
            start_stoch=self.stoch[workers, before],
            start_action=self.actions[workers, before],
        )
        # Without the step before, the sequence starts from a fresh state
        batch["is_first"][:, 0] |= first - 1 < np.maximum(self.added[workers] - self.size, 0)
        # An episode cut off by the next one ends there
        batch["is_last"][:, :-1] |= batch["is_first"][:, 1:]
        return batch

    def _sample_start(self, online: bool) -> tuple[int, int]:
        """A worker and the number of its added steps at which a sequence starts that is still fully stored."""
        while online and self.online:
            worker, start = self.online.popleft()
            if start >= self.added[worker] - self.size:
                return worker, start
        available = np.maximum(np.minimum(self.added, self.size) - self.length + 1, 0)
        worker = int(self.rng.choice(self.num_workers, p=available / available.sum()))
        start = int(self.added[worker] - min(self.added[worker], self.size) + self.rng.integers(available[worker]))
        return worker, start

    def _sample_rare_starts(self, count: int) -> list[tuple[int, int]]:
        """Starts of sequences that each contain a step picked by its rarity, at a random position."""
        if not count:
            return []
        stored = np.minimum(self.added, self.size)
        # Only workers with complete sequences, and of those only the stored steps, which are the first ones
        stored[stored < self.length] = 0
        totals = np.array([self.rarity[worker, :size].sum(dtype=np.float64) for worker, size in enumerate(stored)])
        starts = []
        for worker in self.rng.choice(self.num_workers, count, p=totals / totals.sum()).tolist():
            cumulative = np.cumsum(self.rarity[worker, : stored[worker]], dtype=np.float64)
            index = int(np.searchsorted(cumulative, self.rng.random() * cumulative[-1], side="right"))
            oldest = int(self.added[worker] - stored[worker])
            step = oldest + (index - oldest) % self.size
            start = step - int(self.rng.integers(self.length))
            starts.append((worker, min(max(start, oldest), int(self.added[worker]) - self.length)))
        return starts
