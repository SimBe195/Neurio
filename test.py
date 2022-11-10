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

    base_name = "2022-09-17_17:21_lin-dec_0.0004_24-workers_conv-4x32-3-2_lin-0x256_com-1x256_actval-0x256"

    config.environment.env_name = "SuperMarioBros-v0"
    env = MultiprocessEnvironment(1, config.environment)
    agent = get_agent(
        config.agent,
        1,
        env.observation_space.shape,
        env.action_space.n,
        None,
    )

    checkpoint_path = f"{get_original_cwd()}/models/{base_name}/checkpoint."
    if config.load_step > 0:
        agent.load(checkpoint_path + f"{config.load_step:05d}")

    GameLoop(config, env, agent, None, 1, checkpoint_path).run(
        render=True,
        train=False,
    )


if __name__ == "__main__":
    main()
