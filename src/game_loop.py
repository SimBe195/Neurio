import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

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
        self.current_states, _ = self.env.reset()

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
            steps_per_update = 1000000

        try:
            for u in tqdm(range(num_updates)):
                for _ in range(steps_per_update):
                    self.agent.feed_observation(self.current_states)
                    actions, _ = self.agent.next_actions(train)
                    states, rewards, terminateds, truncateds, metrics = self.env.step(
                        actions
                    )
                    self.current_states = states

                    dones = np.logical_or(terminateds, truncateds)

                    if render:
                        self.env.render()

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

                self.agent.update()
        except KeyboardInterrupt:
            return
