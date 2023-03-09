import logging

import hydra

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)

from omegaconf import DictConfig

from src.agents import get_agent
from src.checkpoints import CheckpointHandler
from src.environment import (
    get_channels,
    get_height,
    get_multiprocess_environment,
    get_num_actions,
    get_stack_frames,
    get_width,
)
from src.game_loop import GameLoop
from src.summary import Summary


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    # Get most recent iter
    level = config.level
    checkpoint_handler = CheckpointHandler(config.experiment_name, level)
    start_iter = 0
    save_path = None
    if checkpoint_handler.checkpoints_exist():
        start_iter = checkpoint_handler.find_max_saved_iter()
        save_path = checkpoint_handler.get_save_path(start_iter)

    # Create summary
    summary = Summary(config.experiment_name)
    summary.load(save_path)

    # Create env
    train_env = get_multiprocess_environment(
        config.num_workers,
        config=config.environment,
        level=level,
        render_mode=None,
    )

    # Set up agent
    agent = get_agent(
        config=config.agent,
        in_width=get_width(train_env),
        in_height=get_height(train_env),
        in_channels=get_channels(train_env),
        in_stack_frames=get_stack_frames(train_env),
        num_workers=config.num_workers,
        num_actions=get_num_actions(train_env),
        summary=summary,
    )
    agent.load(save_path)

    # Run game loop
    log.info(f"Train on level {level}")
    while start_iter < config.num_iters:
        num_iters = min(config.num_iters - start_iter, config.save_frequency)
        log.info(f"Run loop for {num_iters} iters, starting at iter {start_iter}.")
        GameLoop(config, train_env, agent, summary).run(
            num_iters=num_iters,
            train=True,
        )

        start_iter += num_iters
        save_path = checkpoint_handler.get_save_path(start_iter)
        save_path.mkdir(parents=True)
        log.info(f"Save model checkpoint at iter {start_iter}.")
        agent.save(save_path)
        summary.save(save_path)


if __name__ == "__main__":
    main()
