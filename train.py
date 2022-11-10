import logging

import hydra

from src.level_schedule import get_linear_level_schedule

logging.basicConfig(level=logging.ERROR)

from omegaconf import DictConfig

from src.agents import get_agent
from src.checkpoints import CheckpointHandler
from src.environment import (
    get_multiprocess_environment,
    get_num_actions,
    get_singleprocess_environment,
)
from src.game_loop import GameLoop
from src.summary import Summary


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:

    summary = Summary(config.experiment_name)

    agent = get_agent(
        config.agent,
        config.num_workers,
        get_num_actions(config.environment),
        summary,
    )

    for level in get_linear_level_schedule(num_repetitions_per_level=5):
        checkpoint_handler = CheckpointHandler(config.experiment_name, level)
        start_epoch = 0
        if checkpoint_handler.checkpoints_exist():
            start_epoch = checkpoint_handler.find_max_saved_epoch() + 1
            agent.load(checkpoint_handler.get_save_path(start_epoch - 1))

        agent.set_num_workers(config.num_workers)
        logging.info(f"Train on level {level}, starting at epoch {start_epoch}")
        env = get_multiprocess_environment(
            config.num_workers,
            config=config.environment,
            level=level,
            render_mode="human",
        )

        GameLoop(config, env, agent, summary).run(
            render=False,
            train=True,
        )
        env.close()

        agent.save(checkpoint_handler.get_save_path(start_epoch))

        agent.set_num_workers(1)

        logging.info(f"Eval on level {level}")
        env = get_singleprocess_environment(
            config=config.environment,
            level=level,
            recording_path=f"recordings/{config.experiment_name}/{level}",
            video_prefix=f"epoch-{start_epoch}",
            render_mode="rgb_array",
        )

        GameLoop(config, env, agent, None).run(
            render=False,
            train=False,
        )
        env.close()


if __name__ == "__main__":
    main()
