from typing import Optional
from agent import Agent
import logging

from environment import Environment


class GameLoop:
    def __init__(
        self,
        environment: Environment,
        agent: Agent,
    ) -> None:
        self.env = environment
        self.agent = agent

    def reset(self) -> None:
        state = self.env.reset()
        self.agent.reset()
        self.agent.feed_state(state)

    def run_episode(
        self,
        max_steps: Optional[int] = None,
        render: bool = False,
    ) -> None:
        self.reset()

        total_reward = 0
        episode_step = 0

        while max_steps is None or episode_step < max_steps:
            episode_step += 1

            action = self.agent.next_action()
            state, reward, terminated, truncated, metrics = self.env.step(action)
            logging.debug(f" Step {episode_step}: metrics = {metrics}")
            total_reward += reward
            done = terminated or truncated

            if render:
                self.env.render()

            self.agent.give_reward(reward, done)

            if done:
                break

            self.agent.feed_state(state)

        self.agent.update()

        return episode_step, total_reward
