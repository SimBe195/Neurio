from typing import Optional

import numpy as np
from gym import Env
from omegaconf import DictConfig
from tqdm import tqdm

from src.agents import Agent
from src.summary import Summary


class GameLoop:
    def __init__(
        self,
        config: DictConfig,
        environment: Env,
        agent: Agent,
        summary: Optional[Summary] = None,
    ) -> None:
        self.env = environment
        self.agent = agent
        self.summary = summary

        self.steps_per_iter = config.steps_per_iter
        self.current_states = [None] * self.agent.num_workers

    def reset(self) -> None:
        self.current_states, _ = self.env.reset()

    def run(
        self,
        num_iters: int = 1,
        train: bool = True,
    ) -> None:
        self.reset()

        if train:
            steps_per_iter = self.steps_per_iter
        else:
            assert self.agent.num_workers == 1
            steps_per_iter = 1_000_000

        for _ in tqdm(range(num_iters)):
            for _ in range(steps_per_iter):
                self.agent.feed_observation(self.current_states)  # type: ignore
                actions, _ = self.agent.next_actions(train)
                states, rewards, terminateds, truncateds, metrics = self.env.step(
                    actions
                )
                self.current_states = states

                dones = np.logical_or(terminateds, truncateds)

                self.agent.give_reward(rewards, dones)

                for w in range(self.agent.num_workers):
                    if dones[w] and self.summary:
                        self.summary.log_episode_stat(
                            metrics["final_info"][w]["episode"]["l"],
                            "performance/steps",
                        )
                        self.summary.log_episode_stat(
                            metrics["final_info"][w]["episode"]["r"],
                            "performance/reward",
                        )
                        self.summary.log_episode_stat(
                            metrics["final_info"][w]["x_pos"],
                            "performance/distance",
                        )
                        self.summary.next_episode()
                if not train and dones[0]:
                    break

            if train:
                self.agent.update()
