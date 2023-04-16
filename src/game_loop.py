import time
from typing import Dict, List, Optional

import mlflow
import numpy as np
from gym import Env
from omegaconf import DictConfig

from .agents import Agent
from .reward_trackers import RewardTracker


class GameLoop:
    def __init__(
        self,
        config: DictConfig,
        environment: Env,
        agent: Agent,
        reward_trackers: List[RewardTracker] = [],
    ) -> None:
        self.env = environment
        self.agent = agent
        self.steps_per_iter = config.steps_per_iter
        self.reward_trackers = reward_trackers

        self.total_episodes_finished = 0

        self.reset()

    def reset(self) -> None:
        self.current_states, _ = self.env.reset()

    def log_stats(self, metrics: Dict) -> None:
        reward = metrics["episode"]["r"]
        for tracker in self.reward_trackers:
            tracker.record_reward(reward)

        mlflow.log_metric("episode_reward", reward, step=self.total_episodes_finished)
        mlflow.log_metric("episode_length", metrics["episode"]["l"], step=self.total_episodes_finished)
        mlflow.log_metric("episode_x_pos", metrics["x_pos"], step=self.total_episodes_finished)

        self.total_episodes_finished += 1

    def run_single_step(self, train: bool = True) -> bool:
        """
        Run single step for each agent.

        :returns: True if all agents are done.
        """
        self.agent.feed_observation(self.current_states)  # type: ignore
        actions, _ = self.agent.next_actions(train)
        states, rewards, terminateds, truncateds, metrics = self.env.step(actions)
        self.current_states = states
        dones = np.logical_or(terminateds, truncateds)

        self.agent.give_reward(rewards, dones)

        if "final_info" in metrics:
            for w in [w for w, done in enumerate(dones) if done]:
                self.log_stats(metrics["final_info"][w])

        return all(dones)

    def run_train_iter(self):
        for _ in range(self.steps_per_iter):
            self.run_single_step(train=True)
        self.agent.update()

    def run_test_loop(self):
        assert self.agent.num_workers == 1
        while True:
            if self.run_single_step(train=False):
                self.reset()
            time.sleep(1 / 60)
