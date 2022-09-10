import logging

import hydra

logging.basicConfig(level=logging.INFO)

from omegaconf import DictConfig

from src.agents import get_agent
from src.environment import MultiprocessEnvironment
from src.game_loop import GameLoop
from src.summary import Summary


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:

    summary = Summary(config.experiment_name)

    env = MultiprocessEnvironment(config.num_workers, config.environment)
    agent = get_agent(
        config.agent,
        config.num_workers,
        env.observation_space.shape,
        env.action_space.n,
        summary,
    )

    GameLoop(config, env, agent, summary).run(render=True, train=True)


if __name__ == "__main__":
    main()
