import time
from typing import Dict, List

import mlflow
import numpy as np
from gym import Env

from .agents import Agent
from .reward_trackers import RewardTracker


class GameLoop:
    def __init__(
        self,
        environment: Env,
        agent: Agent,
        reward_trackers: Dict[str, RewardTracker] = {},
    ) -> None:
        self.env = environment
        self.agent = agent
        self.reward_trackers = reward_trackers

        self.total_episodes_finished = 0
        self.total_iters_finished = 0

        self.reset()

    def reset(self) -> None:
        self.current_states, _ = self.env.reset()

    def log_episode_stats(self, metrics: Dict) -> None:
        reward = metrics["episode"]["r"]
        for tracker in self.reward_trackers.values():
            tracker.record_reward(reward)

        mlflow.log_metric("episode_reward", reward, step=self.total_episodes_finished)
        mlflow.log_metric(
            "episode_length", metrics["episode"]["l"], step=self.total_episodes_finished
        )
        mlflow.log_metric(
            "episode_x_pos", metrics["x_pos"], step=self.total_episodes_finished
        )

        self.total_episodes_finished += 1

    def log_iter_stats(self) -> None:
        for name, tracker in self.reward_trackers.items():
            mlflow.log_metric(name, tracker.get_value(), step=self.total_iters_finished)
        self.total_iters_finished += 1

    def run_single_step(self, log_stats: bool) -> bool:
        """
        Run single step for each agent.

        :returns: True if all agents are done.
        """
        self.agent.feed_observation(self.current_states)  # type: ignore
        actions, _ = self.agent.next_actions()
        states, rewards, terminateds, truncateds, metrics = self.env.step(actions)
        self.current_states = states
        dones = np.logical_or(terminateds, truncateds)

        self.agent.give_reward(rewards, dones)

        if log_stats and "final_info" in metrics:
            for w in [w for w, done in enumerate(dones) if done]:
                self.log_episode_stats(metrics["final_info"][w])

        return all(dones)

    def run_train_iter(self, steps: int):
        for _ in range(steps):
            self.run_single_step(log_stats=True)
        self.agent.update()
        self.log_iter_stats()

    def run_test_loop(self, framerate: int, max_episodes: int = 3):
        assert self.agent.num_workers == 1
        episode = 0
        while episode < max_episodes:
            start_time = time.time()
            if self.run_single_step(log_stats=False):
                self.reset()
                episode += 1
            elapsed_time = time.time() - start_time
            sleep_time = max(0, 1 / framerate - elapsed_time)
            time.sleep(sleep_time)
