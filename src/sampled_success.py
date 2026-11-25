"""
How often a checkpoint of a run reaches the flag of each of the run's evaluation levels when the agent samples its
actions as in training, without learning. Unlike greedy_progress.py's single go through all levels, this estimates the
success rate of each level on its own: Mario starts each small, the episodes are seeded, and, as in all evaluation,
there are no sticky actions (see evaluation.evaluation_config).

The rates are logged to the run as sampled/success_<level>, with the checkpoint's iteration as step, and printed.

Usage, with the same run and checkpoint selection as test.py:
    uv run src/sampled_success.py test_run_id=<run id> [test_iter=<iteration>]
"""

import logging
import time

import hydra
import mlflow
from mlflow.entities import Metric
from omegaconf import DictConfig
from tqdm import tqdm

from agents import Play
from evaluation import load_evaluation, sampled_success_rate
from game_loop import GameLoop

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

NUM_WORKERS = 32
# Bounds the standard error of each rate by 0.5 / sqrt(1024), i.e. 1.6 percentage points
EPISODES_PER_LEVEL = 1024


@hydra.main(version_base="1.3", config_path="../configs", config_name="neurio_config")
def main(cfg: DictConfig) -> None:
    with load_evaluation(cfg, num_workers=NUM_WORKERS) as evaluation:
        loop = GameLoop(evaluation.env, evaluation.agent, play=Play.SAMPLE)
        levels = evaluation.config.levels.evaluation_levels()
        rates = {level: sampled_success_rate(loop, level, EPISODES_PER_LEVEL) for level in tqdm(levels)}
    timestamp = int(time.time() * 1000)
    metrics = [
        Metric(f"sampled/success_{level}", rate, timestamp, evaluation.iteration) for level, rate in rates.items()
    ]
    mlflow.MlflowClient().log_batch(evaluation.run_id, metrics=metrics)
    print("| Level | Success |\n|---|---|")
    for level, rate in rates.items():
        print(f"| {level} | {rate:.1%} |")
    log.info(f"Mean over {len(rates)} levels: {sum(rates.values()) / len(rates):.1%}")


if __name__ == "__main__":
    main()
