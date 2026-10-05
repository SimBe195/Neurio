import random
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int64, UInt8


@dataclass
class Episode:
    states: UInt8[torch.Tensor, "time channel height width"]
    prev_actions: Int64[torch.Tensor, " time"]
    actions: Int64[torch.Tensor, " time"]
    returns: Float[torch.Tensor, " time"]
    total_reward: float


class SelfImitationMemory:
    """
    Memory of the agent's own best episodes for self-imitation learning (Oh et al., 2018): it records the episode of
    each worker and keeps the `episodes_per_level` episodes with the highest game reward of every level, with the
    discounted return of each step. Levels are kept apart since their rewards are not comparable.
    """

    def __init__(self, num_workers: int, gamma: float, episodes_per_level: int) -> None:
        self.gamma = gamma
        self.episodes_per_level = episodes_per_level
        self._trajectories: list[dict[str, list]] = [self._empty_trajectory() for _ in range(num_workers)]
        self.episodes: dict[str, list[Episode]] = {}
        self._samples: tuple[torch.Tensor, ...] | None = None

    @staticmethod
    def _empty_trajectory() -> dict[str, list]:
        return {"states": [], "prev_actions": [], "actions": [], "rewards": []}

    def record_actions(
        self,
        states: Float[torch.Tensor, "worker channel height width"],
        prev_actions: Int64[torch.Tensor, " worker"],
        actions: Int64[torch.Tensor, " worker"],
    ) -> None:
        # Observations are pixel values, so they are stored compactly as bytes
        states = states.to(torch.uint8)
        for worker, trajectory in enumerate(self._trajectories):
            trajectory["states"].append(states[worker])
            trajectory["prev_actions"].append(prev_actions[worker])
            trajectory["actions"].append(actions[worker])

    def record_rewards(self, rewards: Float[torch.Tensor, " worker"], dones: list[bool], levels: list[str]) -> None:
        """
        :param rewards: The game's reward of the last actions, without exploration bonus
        :param dones: Whether the episode of each worker ended with the last actions
        :param levels: Level of each worker, for finished episodes the level they ended on
        """
        for worker, trajectory in enumerate(self._trajectories):
            trajectory["rewards"].append(float(rewards[worker]))
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
        device = trajectory["states"][0].device
        episodes.append(
            Episode(
                states=torch.stack(trajectory["states"]),
                prev_actions=torch.stack(trajectory["prev_actions"]),
                actions=torch.stack(trajectory["actions"]),
                returns=torch.tensor(returns, dtype=torch.float32, device=device),
                total_reward=total_reward,
            )
        )
        self._samples = None

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
        :return: States, previous actions, actions and returns of random steps of the stored episodes, None if there
            are none
        """
        if self._samples is None:
            stored = [episode for episodes in self.episodes.values() for episode in episodes]
            if not stored:
                return None
            self._samples = tuple(
                torch.cat([getattr(episode, key) for episode in stored])
                for key in ("states", "prev_actions", "actions", "returns")
            )
        states, prev_actions, actions, returns = self._samples
        idx = torch.as_tensor(random.choices(range(len(actions)), k=batch_size), device=actions.device)
        return states[idx], prev_actions[idx], actions[idx], returns[idx]

    def get_metrics(self) -> dict[str, float]:
        stored = [episode for episodes in self.episodes.values() for episode in episodes]
        if not stored:
            return {"sil_stored_episodes": 0.0}
        return {
            "sil_stored_episodes": float(len(stored)),
            "sil_best_episode_reward": max(episode.total_reward for episode in stored),
            "sil_worst_stored_reward": min(episode.total_reward for episode in stored),
        }
