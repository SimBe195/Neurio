"""
How far each checkpoint of a run gets in one greedy go: like test.py, the agent plays the run's evaluation levels in
order, from small Mario in 1-1 on, until the first one without the flag. Its progress is the sum of the distances
its episodes got (see environment.ProgressShaping), i.e. those to the flags of the cleared levels plus the one to where
it failed.

The results are logged to the run as the metrics greedy/progress and greedy/levels_cleared, with the checkpoint's
iteration as step, so that later calls only evaluate the checkpoints saved since. The best checkpoint, the earliest one
with the most progress, is tagged as best_greedy_iter. A plot of the progress over training is stored with the run and
in plots/.

Usage, with the same run selection as test.py:
    uv run src/greedy_progress.py test_run_id=<run id>
"""

import logging
import time
from pathlib import Path

import hydra
import mlflow
from matplotlib.figure import Figure
from mlflow.entities import Metric
from omegaconf import DictConfig
from tqdm import tqdm

from evaluation import checkpoint_iters, load_evaluation, play_levels
from game_loop import GameLoop
from resume import BEST_GREEDY_ITER_TAG

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PROGRESS = "greedy/progress"
LEVELS_CLEARED = "greedy/levels_cleared"

SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def plot(progress: dict[int, float], best: int, levels_cleared: float, title: str, path: Path) -> None:
    iters = sorted(progress)
    figure = Figure(figsize=(10, 4.5), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = figure.add_subplot(facecolor=SURFACE)
    axes.plot(iters, [progress[i] for i in iters], color=SERIES, linewidth=1.5)
    axes.plot(best, progress[best], "o", markersize=8, color=SERIES, markeredgecolor=SURFACE, markeredgewidth=2)
    # The label goes to the side with more room
    left = best > (iters[0] + iters[-1]) / 2
    axes.annotate(
        f"best: iteration {best}, {progress[best]:,.0f} px, {levels_cleared:.0f} levels cleared",
        (best, progress[best]),
        xytext=(-10 if left else 10, 6),
        textcoords="offset points",
        ha="right" if left else "left",
        color=TEXT,
    )
    axes.set_title(title, loc="left", color=TEXT)
    axes.set_xlabel("Training iteration", color=TEXT_SECONDARY)
    axes.set_ylabel("Horizontal progress (px)", color=TEXT_SECONDARY)
    axes.set_ylim(bottom=0)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(TEXT_SECONDARY)
    figure.savefig(path)


@hydra.main(version_base="1.3", config_path="../configs", config_name="neurio_config")
def main(cfg: DictConfig) -> None:
    with load_evaluation(cfg) as evaluation:
        run_id, agent = evaluation.run_id, evaluation.agent
        client = mlflow.MlflowClient()
        progress = {int(m.step): m.value for m in client.get_metric_history(run_id, PROGRESS)}
        new_iters = [i for i in checkpoint_iters(run_id) if i not in progress]
        log.info(f"{len(progress)} checkpoints evaluated before, {len(new_iters)} new ones.")
        loop = GameLoop(evaluation.env, agent, greedy=True)
        levels = evaluation.config.levels.evaluation_levels()
        for iteration in tqdm(new_iters):
            agent.load(run_id, iteration)
            episodes = play_levels(loop, levels)
            progress[iteration] = sum(episode.distance for episode in episodes)
            cleared = sum(episode.flag_get for episode in episodes)
            timestamp = int(time.time() * 1000)
            client.log_batch(
                run_id,
                metrics=[
                    Metric(PROGRESS, progress[iteration], timestamp, iteration),
                    Metric(LEVELS_CLEARED, cleared, timestamp, iteration),
                ],
            )
            log.info(f"Iteration {iteration}: {progress[iteration]} px, {cleared} levels cleared.")

    best = max(sorted(progress), key=progress.__getitem__)
    cleared = {int(m.step): m.value for m in client.get_metric_history(run_id, LEVELS_CLEARED)}[best]
    client.set_tag(run_id, BEST_GREEDY_ITER_TAG, str(best))
    run_name = client.get_run(run_id).info.run_name
    path = Path("plots") / f"{run_name}-greedy-progress.png"
    path.parent.mkdir(exist_ok=True)
    plot(progress, best, cleared, f"Greedy progress of {run_name} per checkpoint", path)
    client.log_artifact(run_id, str(path))
    log.info(f"Best checkpoint: iteration {best} with {progress[best]:,.0f} px. Plot in {path}.")


if __name__ == "__main__":
    main()
