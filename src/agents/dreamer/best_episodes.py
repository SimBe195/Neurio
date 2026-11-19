from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import torch

from .replay import Latent


@dataclass
class Episode:
    frames: npt.NDArray[np.uint8]
    actions: npt.NDArray[np.int64]
    rewards: npt.NDArray[np.float32]
    # Discounted return after each step, without bootstrapping beyond the end of the episode
    returns: npt.NDArray[np.float32]
    deter: npt.NDArray[np.float16]
    stoch: npt.NDArray[np.uint8]
    is_terminal: bool
    game_reward: float

    def __len__(self) -> int:
        return len(self.actions)


class BestEpisodes:
    """
    The agent's own best episodes for self-imitation: the `per_level` ones with the highest game reward of each level,
    since the rewards of different levels are not comparable, nor those of different start power-ups. Steps are recorded
    like in the replay: each frame with the reward received on reaching it, the action taken in it and the model state
    the agent had in it.

    Only episodes of at least `length` steps are kept, so that sequences of that length can be sampled from them.
    """

    def __init__(
        self,
        num_workers: int,
        per_level: int,
        length: int,
        discount: float,
        deter: int,
        stoch: int,
        seed: int | None = None,
    ) -> None:
        """
        :param deter: Size of the deterministic model state
        :param stoch: Number of stochastic variables of the model state
        """
        self.per_level = per_level
        self.length = length
        self.discount = discount
        self.episodes: dict[str, list[Episode]] = {}
        self._current: list[list[tuple[np.ndarray, int, float, float, Latent]]] = [[] for _ in range(num_workers)]
        # The model state of the final frame, in which the agent does not act
        self._no_latent = Latent(np.zeros(deter, np.float16), np.zeros(stoch, np.uint8))
        self.rng = np.random.default_rng(seed)

    def add(
        self, worker: int, frame: np.ndarray, action: int, reward: float, game_reward: float, latent: Latent
    ) -> None:
        self._current[worker].append((np.array(frame), action, reward, game_reward, latent))

    def finish(
        self, worker: int, frame: np.ndarray, reward: float, game_reward: float, is_terminal: bool, level: str
    ) -> None:
        """
        Ends the worker's episode with its final frame, and keeps it if it is among the best of its level.

        :param level: The level, or a group of its episodes that are compared, e.g. "1-2 fire" for a start power-up
        """
        self.add(worker, frame, 0, reward, game_reward, self._no_latent)
        steps, self._current[worker] = self._current[worker], []
        total = float(sum(step[3] for step in steps))
        if len(steps) < self.length:
            return
        episodes = self.episodes.setdefault(level, [])
        if len(episodes) >= self.per_level:
            worst = min(range(len(episodes)), key=lambda i: episodes[i].game_reward)
            if episodes[worst].game_reward >= total:
                return
            episodes.pop(worst)
        rewards = np.array([step[2] for step in steps], np.float32)
        returns = np.zeros_like(rewards)
        for t in reversed(range(len(steps) - 1)):
            returns[t] = rewards[t + 1] + self.discount * returns[t + 1]
        episodes.append(
            Episode(
                frames=np.stack([step[0] for step in steps]),
                actions=np.array([step[1] for step in steps], np.int64),
                rewards=rewards,
                returns=returns,
                deter=np.stack([step[4].deter for step in steps]),
                stoch=np.stack([step[4].stoch for step in steps]),
                is_terminal=is_terminal,
                game_reward=total,
            )
        )

    def __len__(self) -> int:
        return sum(len(episodes) for episodes in self.episodes.values())

    def state_dict(self) -> dict:
        """The stored episodes, without those in progress."""

        def tensors(episode: Episode) -> dict:
            # Plain Python values, since loading the state safely refuses NumPy's scalars
            return {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else np.asarray(v).item()
                for k, v in asdict(episode).items()
            }

        return {level: [tensors(episode) for episode in episodes] for level, episodes in self.episodes.items()}

    def load_state_dict(self, state: dict) -> None:
        self.episodes = {
            level: [
                Episode(**{k: v.numpy().copy() if isinstance(v, torch.Tensor) else v for k, v in episode.items()})
                for episode in episodes
            ]
            for level, episodes in state.items()
        }

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        :return: Sequences like those of Replay.sample, from uniformly sampled steps of the stored episodes, with
            "returns" in addition. Those at the start of an episode start from a fresh state.
        """
        stored = [episode for episodes in self.episodes.values() for episode in episodes]
        assert stored, "No episodes stored yet."
        starts = np.array([len(episode) - self.length + 1 for episode in stored])
        chosen = self.rng.choice(len(stored), batch_size, p=starts / starts.sum())
        windows = [(stored[i], int(self.rng.integers(starts[i]))) for i in chosen]
        batch = {
            name: np.stack([getattr(episode, name)[start : start + self.length] for episode, start in windows])
            for name in ("frames", "actions", "rewards", "returns")
        }
        is_last = np.zeros((batch_size, self.length), bool)
        for row, (episode, start) in enumerate(windows):
            is_last[row, -1] = start + self.length == len(episode)
        is_first = np.zeros_like(is_last)
        is_first[:, 0] = [start == 0 for _, start in windows]
        terminal = np.array([episode.is_terminal for episode, _ in windows])
        # The step before each sequence, at the start of an episode its first one, which is_first resets
        before = [(episode, max(start - 1, 0)) for episode, start in windows]
        return {
            **batch,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_last & terminal[:, None],
            "start_deter": np.stack([episode.deter[index] for episode, index in before]),
            "start_stoch": np.stack([episode.stoch[index] for episode, index in before]),
            "start_action": np.array([episode.actions[index] for episode, index in before]),
        }
