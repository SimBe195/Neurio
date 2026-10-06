import bisect
import itertools
import random
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int64, UInt8


@dataclass
class Episode:
    # Each frame once, the state of step t is frames t to t + stack_frames - 1
    frames: UInt8[torch.Tensor, "frame channel height width"]
    prev_actions: Int64[torch.Tensor, " time"]
    actions: Int64[torch.Tensor, " time"]
    returns: Float[torch.Tensor, " time"]
    total_reward: float


class SelfImitationMemory:
    """
    Memory of the agent's own best episodes for self-imitation learning (Oh et al., 2018): it records the episode of
    each worker and keeps the `episodes_per_level` episodes with the highest game reward of every level, with the
    discounted return of each step. Levels are kept apart since their rewards are not comparable.

    Episodes are kept in CPU memory. States are stacks of the last `stack_frames` frames, so consecutive states share
    all but one frame, and each frame is stored only once.
    """

    def __init__(self, num_workers: int, gamma: float, episodes_per_level: int, stack_frames: int) -> None:
        self.gamma = gamma
        self.episodes_per_level = episodes_per_level
        self.stack_frames = stack_frames
        self._trajectories: list[dict[str, list]] = [self._empty_trajectory() for _ in range(num_workers)]
        self.episodes: dict[str, list[Episode]] = {}
        self._stored: list[Episode] | None = None
        self._cumulative_lengths: list[int] = []

    @staticmethod
    def _empty_trajectory() -> dict[str, list]:
        return {"frames": [], "prev_actions": [], "actions": [], "rewards": []}

    def record_actions(
        self,
        states: Float[torch.Tensor, "worker channel height width"],
        prev_actions: Int64[torch.Tensor, " worker"],
        actions: Int64[torch.Tensor, " worker"],
    ) -> None:
        # Observations are pixel values, so they are stored compactly as bytes
        frames = states.to("cpu", torch.uint8).unflatten(1, (self.stack_frames, -1))
        for worker, (trajectory, prev_action, action) in enumerate(
            zip(self._trajectories, prev_actions.tolist(), actions.tolist(), strict=True)
        ):
            # The first state of an episode contributes all its frames, the following ones only their newest. They are
            # copied, since a view would keep the frames of all workers alive.
            new_frames = frames[worker] if not trajectory["actions"] else frames[worker, -1:]
            trajectory["frames"].extend(new_frames.clone())
            trajectory["prev_actions"].append(prev_action)
            trajectory["actions"].append(action)

    def record_rewards(self, rewards: Float[torch.Tensor, " worker"], dones: list[bool], levels: list[str]) -> None:
        """
        :param rewards: The game's reward of the last actions, without exploration bonus
        :param dones: Whether the episode of each worker ended with the last actions
        :param levels: Level of each worker, for finished episodes the level they ended on
        """
        for worker, (trajectory, reward) in enumerate(zip(self._trajectories, rewards.tolist(), strict=True)):
            trajectory["rewards"].append(reward)
            if dones[worker]:
                self._store(trajectory, levels[worker])
                self._trajectories[worker] = self._empty_trajectory()

    def _store(self, trajectory: dict[str, list], level: str) -> None:
        if not trajectory["actions"] or len(trajectory["actions"]) != len(trajectory["rewards"]):
            return
        total_reward = sum(trajectory["rewards"])
        episodes = self.episodes.setdefault(level, [])
        if len(episodes) >= self.episodes_per_level:
            worst = min(range(len(episodes)), key=lambda i: episodes[i].total_reward)
            if episodes[worst].total_reward >= total_reward:
                return
            episodes.pop(worst)

        returns, discounted = [], 0.0
        for reward in reversed(trajectory["rewards"]):
            discounted = reward + self.gamma * discounted
            returns.append(discounted)
        returns.reverse()
        episodes.append(
            Episode(
                frames=torch.stack(trajectory["frames"]),
                prev_actions=torch.tensor(trajectory["prev_actions"]),
                actions=torch.tensor(trajectory["actions"]),
                returns=torch.tensor(returns, dtype=torch.float32),
                total_reward=total_reward,
            )
        )
        self._stored = None

    def sample(
        self, batch_size: int
    ) -> (
        tuple[
            UInt8[torch.Tensor, "batch channel height width"],
            Int64[torch.Tensor, " batch"],
            Int64[torch.Tensor, " batch"],
            Float[torch.Tensor, " batch"],
        ]
        | None
    ):
        """
        :return: States, previous actions, actions and returns of random steps of the stored episodes, on the CPU, None
            if there are none
        """
        if self._stored is None:
            self._stored = [episode for episodes in self.episodes.values() for episode in episodes]
            self._cumulative_lengths = list(itertools.accumulate(len(episode.actions) for episode in self._stored))
        if not self._stored:
            return None
        states, prev_actions, actions, returns = [], [], [], []
        for step in random.choices(range(self._cumulative_lengths[-1]), k=batch_size):
            i = bisect.bisect_right(self._cumulative_lengths, step)
            episode = self._stored[i]
            t = step - (self._cumulative_lengths[i - 1] if i > 0 else 0)
            states.append(episode.frames[t : t + self.stack_frames].flatten(0, 1))
            prev_actions.append(episode.prev_actions[t])
            actions.append(episode.actions[t])
            returns.append(episode.returns[t])
        return torch.stack(states), torch.stack(prev_actions), torch.stack(actions), torch.stack(returns)

    def state_dict(self) -> dict:
        """The stored episodes. Unfinished episodes are not kept, since the environments restart."""
        return {level: [vars(episode) for episode in episodes] for level, episodes in self.episodes.items()}

    def load_state_dict(self, state: dict) -> None:
        self.episodes = {level: [Episode(**episode) for episode in episodes] for level, episodes in state.items()}
        self._stored = None

    def get_metrics(self) -> dict[str, float]:
        stored = [episode for episodes in self.episodes.values() for episode in episodes]
        if not stored:
            return {"sil_stored_episodes": 0.0}
        return {
            "sil_stored_episodes": float(len(stored)),
            "sil_stored_steps": float(sum(len(episode.actions) for episode in stored)),
            "sil_best_episode_reward": max(episode.total_reward for episode in stored),
            "sil_worst_stored_reward": min(episode.total_reward for episode in stored),
        }
