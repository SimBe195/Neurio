import logging

logging.basicConfig(level=logging.INFO)
from environment import Environment, SubsamplingWrapper
from agent import PPOAgent, RandomAgent
from reward_func import RewardWrapperV1
from game_loop import GameLoop


if __name__ == "__main__":
    env = Environment()
    env = SubsamplingWrapper(env, num_steps=4, new_width=84, new_height=84)
    env = RewardWrapperV1(env)
    # agent = RandomAgent(environment=env, train_mode=True)
    agent = PPOAgent(environment=env, train_mode=True)

    loop = GameLoop(env, agent)
    for episode in range(1000):
        steps, reward = loop.run_episode(min((episode // 50 + 1) * 100, 1000), True)
        logging.info(
            f" Episode {episode+1} finished after {steps} steps. Total reward: {reward}."
        )
