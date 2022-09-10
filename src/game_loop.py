import logging

import numpy as np
from omegaconf import DictConfig

from src.agents import Agent
from src.environment import BaseEnvironment
from src.summary import Summary


class GameLoop:
    def __init__(
        self,
        config: DictConfig,
        environment: BaseEnvironment,
        agent: Agent,
        summary: Summary,
    ) -> None:
        self.env = environment
        self.agent = agent
        self.summary = summary

        self.num_updates = config.num_updates
        self.steps_per_update = config.steps_per_update
        self.current_states = [None] * self.agent.num_workers

    def reset(self) -> None:
        self.current_states = self.env.reset()

    def run(
        self,
        render: bool = False,
        train: bool = True,
    ) -> None:
        self.reset()

        if train:
            num_updates = self.num_updates
            steps_per_update = self.steps_per_update
        else:
            num_updates = 1
            steps_per_update = self.float("inf")

        episode_rewards = [0] * self.agent.num_workers
        episode_steps = [0] * self.agent.num_workers
        for _ in range(num_updates):
            for _ in range(steps_per_update):
                self.agent.feed_observation(self.current_states)
                actions = self.agent.next_actions(train)
                states, rewards, terminateds, truncateds, metrics = self.env.step(
                    actions
                )
                self.current_states = states
                for w, (m, r) in enumerate(zip(metrics, rewards)):
                    logging.debug(f"  Worker {w}, step {episode_steps[w]}: {m}")
                    episode_steps[w] += 1
                    episode_rewards[w] += r

                dones = np.logical_or(terminateds, truncateds)

                if render:
                    self.env.render()

                self.agent.give_reward(rewards, dones)

                for w in range(self.agent.num_workers):
                    if dones[w]:
                        logging.info(
                            f"Worker {w} finished episode. Steps: {episode_steps[w]}, reward: {episode_rewards[w]}, metrics: {metrics[w]}"
                        )
                        self.summary.log_episode_stat(episode_steps[w], "steps")
                        self.summary.log_episode_stat(episode_rewards[w], "reward")
                        self.summary.log_episode_stat(metrics[w]["x_pos"], "distance")
                        episode_steps[w] = 0
                        episode_rewards[w] = 0
                        self.summary.next_episode()
                        if not train:
                            break
            if train:
                self.agent.update()
