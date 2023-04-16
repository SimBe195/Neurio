import time
from typing import Dict, Optional

import numpy as np
from gym import Env
from omegaconf import DictConfig

from src.agents import Agent
from src.moving_average import MovingAverage


class GameLoop:
    def __init__(
        self,
        config: DictConfig,
        environment: Env,
        agent: Agent,
        reward_tracker: Optional[MovingAverage] = None,
    ) -> None:
        self.env = environment
        self.agent = agent
        self.steps_per_iter = config.steps_per_iter
        self.reward_tracker = reward_tracker

        self.reset()

    def reset(self) -> None:
        self.current_states, _ = self.env.reset()

    def log_stats(self, metrics: Dict) -> None:
        reward = metrics["episode"]["r"]
        if self.reward_tracker:
            self.reward_tracker.record_data(reward)

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
                break
            time.sleep(1 / 60)
