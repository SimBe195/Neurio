from collections import deque
from dataclasses import dataclass

import mlflow
import numpy as np
from gymnasium.vector import AsyncVectorEnv

from agents import Agent, StepOutcome
from config.levels import level_index
from level_schedulers import LevelScheduler
from novelty import NoveltyBonus


@dataclass
class EpisodeStats:
    reward: float
    length: int
    level: str
    x_pos: int
    flag_get: bool


class GameLoop:
    def __init__(
        self,
        environment: AsyncVectorEnv,
        agent: Agent,
        level_scheduler: LevelScheduler | None = None,
        objective_window: int = 100,
        novelty_bonus: NoveltyBonus | None = None,
    ) -> None:
        """
        :param objective_window: Number of most recent episodes whose mean reward is reported
        :param novelty_bonus: Exploration bonus for the agent to learn from; episode rewards stay the game's own
        """
        self.env = environment
        self.agent = agent
        self.level_scheduler = level_scheduler
        self.novelty_bonus = novelty_bonus

        self.iter_episodes: list[EpisodeStats] = []
        self.iter_novelty_bonus = 0.0
        self.recent_rewards: deque[float] = deque(maxlen=objective_window)
        self.total_episodes_finished = 0
        self.total_iters_finished = 0
        self.total_env_steps = 0

        self.env.reset_async()
        self.current_states: np.ndarray = self.env.reset_wait()[0]

    @property
    def recent_mean_reward(self) -> float | None:
        """Mean reward of the most recent episodes, None before the first episode finished."""
        if not self.recent_rewards:
            return None
        return float(np.mean(self.recent_rewards))

    def run_single_step(self) -> None:
        self.agent.feed_observation(self.current_states)
        actions, _ = self.agent.next_actions()
        self.env.step_async(np.array(actions))
        self.current_states, rewards, terminated, truncated, infos = self.env.step_wait()
        self.total_env_steps += len(actions)
        dones = np.logical_or(terminated, truncated)

        # Finished episodes report their last step in final_info, the regular info already belongs to the next episode
        step_infos = [infos["final_info"] if done else infos for done in dones]
        levels = [f"{int(info['world'][w])}-{int(info['stage'][w])}" for w, info in enumerate(step_infos)]
        bonus = None
        if self.novelty_bonus is not None:
            positions = [(levels[w], int(info["x_pos"][w]), int(info["y_pos"][w])) for w, info in enumerate(step_infos)]
            bonus = self.novelty_bonus(positions)
            self.iter_novelty_bonus += float(bonus.sum())

        self.agent.give_reward(StepOutcome(rewards, terminated, truncated, levels, infos.get("final_obs"), bonus))
        for worker in np.flatnonzero(dones):
            self.record_episode(infos["final_info"], int(worker), levels[worker])

    def record_episode(self, info: dict, worker: int, level: str) -> None:
        episode = EpisodeStats(
            reward=float(info["episode"]["r"][worker]),
            length=int(info["episode"]["l"][worker]),
            level=level,
            x_pos=int(info["x_pos"][worker]),
            flag_get=bool(info["flag_get"][worker]),
        )
        if self.level_scheduler is not None:
            self.level_scheduler.record_episode(level, episode.flag_get)
        self.iter_episodes.append(episode)
        self.recent_rewards.append(episode.reward)
        self.total_episodes_finished += 1

    def get_episode_metrics(self) -> dict[str, float]:
        """Aggregated stats of the episodes finished in the current iteration."""
        metrics: dict[str, float] = {
            "episodes": len(self.iter_episodes),
            "total_episodes": self.total_episodes_finished,
        }
        if (recent_mean_reward := self.recent_mean_reward) is not None:
            metrics["recent_mean_reward"] = recent_mean_reward
        if not self.iter_episodes:
            return metrics
        rewards = [e.reward for e in self.iter_episodes]
        x_positions = [e.x_pos for e in self.iter_episodes]
        # The level where an episode ended, which for the full game is the furthest one reached
        level_indices = [level_index(e.level) for e in self.iter_episodes]
        metrics.update(
            episode_reward_mean=float(np.mean(rewards)),
            episode_reward_max=float(np.max(rewards)),
            episode_length_mean=float(np.mean([e.length for e in self.iter_episodes])),
            episode_x_pos_mean=float(np.mean(x_positions)),
            episode_x_pos_max=float(np.max(x_positions)),
            episode_level_index_mean=float(np.mean(level_indices)),
            episode_level_index_max=float(np.max(level_indices)),
            flag_rate=float(np.mean([e.flag_get for e in self.iter_episodes])),
        )
        return metrics

    def update_levels(self) -> None:
        if self.level_scheduler is None or not self.level_scheduler.update():
            return
        distribution = self.level_scheduler.distribution()
        assert distribution is not None
        self.env.call("set_level_distribution", *distribution)

    def run_train_iter(self, steps: int) -> dict[str, float]:
        """
        Collect experience for `steps` steps per worker, update the agent and log all metrics of the iteration in one
        batch, with the total number of environment steps as step.

        :return: The logged metrics
        """
        self.iter_episodes = []
        self.iter_novelty_bonus = 0.0
        for _ in range(steps):
            self.run_single_step()

        episode_metrics = self.get_episode_metrics()
        train_metrics = self.agent.update(self.current_states, episode_metrics.get("episode_reward_mean"))
        self.update_levels()

        metrics = {**episode_metrics, **train_metrics, "iteration": self.total_iters_finished}
        if self.level_scheduler is not None:
            metrics.update(self.level_scheduler.get_metrics())
        if self.novelty_bonus is not None:
            metrics["novelty_bonus_per_step"] = self.iter_novelty_bonus / (steps * self.env.num_envs)
            metrics["visited_cells"] = self.novelty_bonus.visited_cells
        mlflow.log_metrics(metrics, step=self.total_env_steps)
        self.total_iters_finished += 1
        return metrics

    def run_test_episode(self) -> EpisodeStats:
        """Play one episode without training. With render mode "human", the environment shows it in real time."""
        assert self.agent.num_workers == 1
        self.iter_episodes = []
        while not self.iter_episodes:
            self.run_single_step()
        return self.iter_episodes[0]
