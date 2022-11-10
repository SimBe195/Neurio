import logging

import hydra

from src.level_schedule import get_linear_level_schedule

logging.basicConfig(level=logging.ERROR)

from omegaconf import DictConfig

from src.agents import get_agent
from src.checkpoints import CheckpointHandler
from src.environment import get_num_actions, get_singleprocess_environment
from src.game_loop import GameLoop
from src.summary import Summary


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:

    agent = get_agent(
        config.agent,
        config.num_workers,
        get_num_actions(config.environment),
        None,
    )

    for level in get_linear_level_schedule(num_repetitions_per_level=1):
        checkpoint_handler = CheckpointHandler(config.experiment_name, level)
        if not checkpoint_handler.checkpoints_exist():
            continue

        start_epoch = checkpoint_handler.find_max_saved_epoch() + 1
        agent.load(checkpoint_handler.get_save_path(start_epoch - 1))

        agent.set_num_workers(1)

        logging.info(f"Eval trained model at epoch {start_epoch} on level {level}")
        env = get_singleprocess_environment(
            config=config.environment,
            level=level,
        )

        GameLoop(config, env, agent, None).run(
            render=True,
            train=False,
        )
