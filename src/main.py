from datetime import datetime
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
from environment import Environment, SubsamplingWrapper
from agent import PPOAgent, RandomAgent
from reward_func import RewardWrapperV1
from game_loop import GameLoop


if __name__ == "__main__":
    train_log_dir = "logs/" + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/train"
    test_log_dir = "logs/" + datetime.now().strftime("%Y-%m-%d_%H-%M") + "/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    env = Environment()
    env = RewardWrapperV1(env)
    env = SubsamplingWrapper(
        env,
        num_steps=4,
        new_width=114,
        new_height=94,
        clip=(33, 19, 28, None),
        greyscale=False,
    )
    # agent = RandomAgent(environment=env, train_mode=True)
    agent = PPOAgent(environment=env, train_mode=True)

    loop = GameLoop(env, agent)
    episode = 0
    for iter in range(1000):
        for episode_iter in range(10):
            episode += 1
            agent.train_mode = True
            steps, reward = loop.run_episode((iter + 1) * 10, False)
            logging.info(
                f" Train episode {episode+1} finished after {steps} steps. Total reward: {reward}."
            )
            with train_summary_writer.as_default():
                tf.summary.scalar("steps", reward, episode)
                tf.summary.scalar("reward", reward, episode)

        agent.train_mode = False
        steps, reward = loop.run_episode(None, True)
        logging.info(
            f" Test episode {iter + 1} finished after {steps} steps. Total reward: {reward}."
        )
        with test_summary_writer.as_default():
            tf.summary.scalar("steps", reward, iter)
            tf.summary.scalar("reward", reward, iter)
