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
        self.total_reward = 0

    def reset(self) -> None:
        state = self.env.reset()
        self.agent.reset()
        self.agent.feed_state(state)
        self.total_reward = 0

    def run(
        self,
        allow_reset: bool = True,
        num_steps: Optional[int] = None,
        render: bool = False,
    ) -> None:
        assert (
            not allow_reset or num_steps
        ), "Unlimited steps plus allowing resets results in an infinite game-loop."

        self.reset()
        global_step = 0
        episode_step = 0
        max_episode_steps = 200
        episode_counter = 0

        while num_steps is None or global_step < num_steps:
            global_step += 1
            episode_step += 1
            action = self.agent.next_action()
            state, reward, terminated, truncated, metrics = self.env.step(action)
            self.total_reward += reward
            done = terminated or truncated
            self.agent.give_reward(reward, done)

            if done or episode_step == max_episode_steps:
                episode_counter += 1
                logging.info(
                    f" Episode {episode_counter} finished. Total reward: {self.total_reward}, final metrics: {metrics}"
                )
                episode_step = 0
                self.agent.update()
                if not allow_reset:
                    break
                self.reset()
            else:
                self.agent.feed_state(state)
            if render:
                self.env.render()
        logging.info(
            f" Run finished. Total reward: {self.total_reward}, final metrics: {metrics}"
        )
        self.reset()
