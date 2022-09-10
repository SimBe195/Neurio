import logging

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
        self.current_state = None

    def reset(self) -> None:
        self.current_state = self.env.reset()

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

        episode_reward = 0
        episode_step = 0
        for _ in range(num_updates):
            for _ in range(steps_per_update):
                self.agent.feed_observation(self.current_state)
                action = self.agent.next_action(train)
                state, reward, terminated, truncated, metrics = self.env.step(action)
                self.current_state = state
                logging.debug(f"Step {episode_step}: metrics = {metrics}")
                episode_step += 1
                episode_reward += reward
                done = terminated or truncated

                if render:
                    self.env.render()

                self.agent.give_reward(reward, done)

                if done:
                    logging.info(
                        f"Episode finished. Steps: {episode_step}, reward: {episode_reward}, metrics: {metrics}"
                    )
                    self.reset()
                    self.summary.log_episode_stat(episode_step, "steps")
                    self.summary.log_episode_stat(episode_reward, "reward")
                    self.summary.log_episode_stat(metrics["x_pos"], "distance")
                    episode_step = 0
                    episode_reward = 0
                    self.summary.next_episode()
                    if not train:
                        break
            if train:
                self.agent.update()
