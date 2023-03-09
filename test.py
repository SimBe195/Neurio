import logging

import hydra

logging.basicConfig(level=logging.ERROR)

log = logging.getLogger(__name__)

from omegaconf import DictConfig

from src.agents import get_agent
from src.checkpoints import CheckpointHandler
from src.environment import get_num_actions, get_singleprocess_environment
from src.game_loop import GameLoop


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    # Set up agent
    agent = get_agent(
        config.agent,
        in_width=114,
        in_height=94,
        in_channels=1,
        in_stack_frames=3,
        num_workers=1,
        num_actions=get_num_actions(config.environment),
    )

    # Load checkpoint
    level = config.level
    log.info(f"Eval trained model on level {level}")

    checkpoint_handler = CheckpointHandler(config.experiment_name, level)
    if not checkpoint_handler.checkpoints_exist():
        log.error("No checkpoint found.")
        return

    load_iter = checkpoint_handler.find_max_saved_iter()
    log.info(f"Loading checkpoint from epoch {load_iter}")
    agent.load(checkpoint_handler.get_save_path(load_iter))

    env = get_singleprocess_environment(
        config=config.environment,
        level=level,
        render_mode="human",
    )

    GameLoop(config, env, agent).run(train=False)


if __name__ == "__main__":
    main()
