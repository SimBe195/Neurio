"""
Deletes the checkpoints of a run that greedy_progress.py has evaluated, except those worth keeping: each one that
cleared more levels in its greedy go than every checkpoint before it and those the run's tags name as the best
(best_checkpoint_iter, best_greedy_iter). Checkpoints not evaluated yet are kept, so that greedy_progress.py can still
evaluate them. Resuming does not need checkpoints, see resume.py. Dry run unless --apply is given.

Usage:
    uv run src/clean_checkpoints.py <run id>            # list what would be deleted
    uv run src/clean_checkpoints.py <run id> --apply    # delete it
"""

import argparse
import shutil

import mlflow

from agents import checkpoint_name
from evaluation import checkpoint_iters
from greedy_progress import LEVELS_CLEARED
from resume import BEST_CHECKPOINT_ITER_TAG, BEST_GREEDY_ITER_TAG, local_artifacts

# Tags of a run that name checkpoints to keep
BEST_TAGS = (BEST_CHECKPOINT_ITER_TAG, BEST_GREEDY_ITER_TAG)


def checkpoints_to_delete(iters: list[int], levels_cleared: dict[int, float], keep: set[int]) -> list[int]:
    """
    :param iters: Iterations of the stored checkpoints
    :param levels_cleared: Levels each evaluated checkpoint cleared in its greedy go, also of those deleted before
    :param keep: Iterations to keep in any case
    """
    records, best = set(), float("-inf")
    for iteration in sorted(levels_cleared):
        if levels_cleared[iteration] > best:
            records.add(iteration)
            best = levels_cleared[iteration]
    return [i for i in sorted(iters) if i in levels_cleared and i not in records and i not in keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id")
    parser.add_argument("--apply", action="store_true", help="Delete the checkpoints instead of listing them")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    tags = client.get_run(args.run_id).data.tags
    keep = {int(tags[tag]) for tag in BEST_TAGS if tag in tags}
    levels_cleared = {int(m.step): m.value for m in client.get_metric_history(args.run_id, LEVELS_CLEARED)}
    iters = checkpoint_iters(args.run_id)
    delete = checkpoints_to_delete(iters, levels_cleared, keep)
    directory = local_artifacts(args.run_id)
    size = sum(f.stat().st_size for i in delete for f in (directory / checkpoint_name(i)).rglob("*") if f.is_file())
    kept = sorted(set(iters) - set(delete))
    print(f"{len(iters)} checkpoints, {len(levels_cleared)} evaluated. Keeping {len(kept)}: {kept}")
    print(f"{'Deleting' if args.apply else 'Would delete'} {len(delete)} checkpoints, {size / 1e9:.1f} GB.")
    if args.apply:
        for iteration in delete:
            shutil.rmtree(directory / checkpoint_name(iteration))


if __name__ == "__main__":
    main()
