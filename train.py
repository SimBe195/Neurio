import logging
from datetime import datetime

import hydra
from hydra.utils import get_original_cwd

from src.level_schedule import get_level_schedule

logging.basicConfig(level=logging.ERROR)

from omegaconf import DictConfig

from src.agents import get_agent
from src.environment import (
    get_multiprocess_environment,
    get_num_actions,
    get_singleprocess_environment,
)
from src.game_loop import GameLoop
from src.summary import Summary


@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:

    base_name = f"{datetime.now():%Y-%m-%d_%H:%M}_{config.experiment_name}"
    summary = Summary(base_name)

    agent = get_agent(
        config.agent,
        config.num_workers,
        get_num_actions(config.environment),
        summary,
    )

    save_checkpoint_path = f"{get_original_cwd()}/models/{base_name}/checkpoint."
    if config.load_name:
        load_checkpoint_path = f"{get_original_cwd()}/models/{config.load_name}/checkpoint.{config.load_iter:03d}"
        logging.info(f"Loading model weights from {load_checkpoint_path}")
        agent.load(load_checkpoint_path)

    for iter, level in enumerate(
        get_level_schedule()[config.load_iter :], start=config.load_iter
    ):
        agent.set_num_workers(config.num_workers)
        logging.info(f"Train on level {level}")
        env = get_multiprocess_environment(
            config.num_workers,
            config=config.environment,
            level=level,
        )

        GameLoop(config, env, agent, summary).run(
            render=False,
            train=True,
        )
        env.close()
        agent.save(save_checkpoint_path + f"{iter:03d}")

        agent.set_num_workers(1)
        if level:
            logging.info(f"Eval on level {level}")
            env = get_singleprocess_environment(
                config=config.environment,
                level=level,
                recording_path=f"recordings/{base_name}/",
                video_prefix=f"iter-{iter+1}_lev-{level}",
            )

            GameLoop(config, env, agent, None).run(
                render=False,
                train=False,
            )

        logging.info(f"Eval whole game")
        env = get_singleprocess_environment(
            config=config.environment,
            recording_path=f"recordings/{base_name}/",
            video_prefix=f"iter-{iter+1}",
        )

        GameLoop(config, env, agent, None).run(
            render=False,
            train=False,
        )


if __name__ == "__main__":
    main()
