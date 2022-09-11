import logging
from datetime import datetime

import hydra
from hydra.utils import get_original_cwd

logging.basicConfig(level=logging.INFO)

from omegaconf import DictConfig

from src.agents import get_agent
from src.environment import MultiprocessEnvironment
from src.game_loop import GameLoop
from src.summary import Summary


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:

    base_name = f"{config.experiment_name}_{datetime.now():%Y-%m-%d_%H:%M}"
    summary = Summary(base_name)

    env = MultiprocessEnvironment(config.num_workers, config.environment)
    agent = get_agent(
        config.agent,
        config.num_workers,
        env.observation_space.shape,
        env.action_space.n,
        summary,
    )

    checkpoint_path = f"{get_original_cwd()}/models/{base_name}/checkpoint."
    if config.load_step > 0:
        agent.load(checkpoint_path + config.load_step)

    GameLoop(config, env, agent, summary, config.save_interval, checkpoint_path).run(
        render=False,
        train=True,
    )


if __name__ == "__main__":
    main()
