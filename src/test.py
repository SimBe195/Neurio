import logging

import hydra
from omegaconf import DictConfig

from agents import Play
from evaluation import load_evaluation, play_levels
from game_loop import GameLoop

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="neurio_config")
def main(cfg: DictConfig) -> None:
    """
    Watch a trained agent play its evaluation levels (see LevelConfig.evaluation_levels) in order, by default those it
    was trained on, e.g. all 32 for the curriculum. Mario starts small and keeps the power-up he reached each flag with
    for the next level, like in the game. It stops at the first level without the flag. Levels given on the command
    line replace them, e.g. `levels=single levels.level=1-3`. The agent always takes its most likely action and the
    episodes are seeded, so they are deterministic.

    Plays the best run of the experiment of the configured level mode, or `test_run_id`, at the checkpoint `test_iter`,
    by default the latest one.
    """
    with load_evaluation(cfg, render_mode="human") as evaluation:
        loop = GameLoop(evaluation.env, evaluation.agent, play=Play.GREEDY)
        for episode in play_levels(loop, evaluation.config.levels.evaluation_levels()):
            flag = "with" if episode.flag_get else "without"
            log.info(f"{episode.level}: ended at x {episode.x_pos} after {episode.distance} px {flag} the flag.")


if __name__ == "__main__":
    main()
