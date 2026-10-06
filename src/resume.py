"""
Resuming interrupted training runs. With every checkpoint, a run stores the full state of training: the agent with its
optimizer and self-imitation memory, the curriculum, the curiosity counts and the progress counters. Runs are tagged
with a hash of their config, so that starting training with the same config again can offer to resume them.
"""

import datetime
import hashlib
import json
import logging
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Settings that do not change how the agent is trained
NON_TRAINING_KEYS = ("tags", "render", "resume", "test_run_id", "test_iter", "tracking_uri", "save_frequency")
CONFIG_HASH_TAG = "config_hash"
RESUME_ITER_TAG = "resume_iter"
RESUME_ARTIFACT_TAG = "resume_artifact"
# The state alternates between two files, so that an interrupted upload leaves the previous state intact
RESUME_ARTIFACTS = ("resume_state_a.pt", "resume_state_b.pt")


def config_hash(cfg: DictConfig) -> str:
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    relevant = {key: value for key, value in container.items() if key not in NON_TRAINING_KEYS}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class ResumableRun:
    run_id: str
    run_name: str | None
    start_time: datetime.datetime
    iteration: int
    num_iters: int


def find_resumable_run(experiment_id: str, cfg_hash: str, num_iters: int) -> ResumableRun | None:
    """The most recently started run with the given config hash that stopped before its last iteration."""
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.{CONFIG_HASH_TAG} = '{cfg_hash}'",
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    for run in runs:
        tags = run.data.tags
        # Runs stopped on purpose, e.g. pruned trials of a hyperparameter search, are not resumed
        if RESUME_ITER_TAG in tags and "stopped_early" not in tags and int(tags[RESUME_ITER_TAG]) < num_iters:
            start_time = datetime.datetime.fromtimestamp(run.info.start_time / 1000)
            return ResumableRun(run.info.run_id, run.info.run_name, start_time, int(tags[RESUME_ITER_TAG]), num_iters)
    return None


def decide_resume(run: ResumableRun | None, resume: bool | None, ask: Callable[[str], str] = input) -> str | None:
    """
    :param resume: Whether to resume an interrupted run, None to ask
    :return: The id of the run to resume, None to start a new one
    """
    if run is None:
        if resume:
            log.info("No interrupted run with this config, starting a new one.")
        return None
    if resume is not None:
        return run.run_id if resume else None

    description = (
        f"Run {run.run_name} ({run.run_id}) with the same config, started {run.start_time:%Y-%m-%d %H:%M}, "
        f"stopped after iteration {run.iteration} of {run.num_iters}."
    )
    if not sys.stdin.isatty():
        raise SystemExit(f"{description} Set resume=true to continue it or resume=false to start a new run.")
    while (answer := ask(f"{description}\nResume it? [y/n] ").strip().lower()) not in ("y", "n"):
        pass
    return run.run_id if answer == "y" else None


def save_training_state(iteration: int, state: dict) -> None:
    """Store the state of training after the given iteration in the active run."""
    active_run = mlflow.active_run()
    assert active_run is not None
    run_id = active_run.info.run_id
    previous = mlflow.MlflowClient().get_run(run_id).data.tags.get(RESUME_ARTIFACT_TAG)
    name = RESUME_ARTIFACTS[1] if previous == RESUME_ARTIFACTS[0] else RESUME_ARTIFACTS[0]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / name
        torch.save({"iteration": iteration, **state}, path)
        mlflow.log_artifact(str(path))
    mlflow.set_tags({RESUME_ARTIFACT_TAG: name, RESUME_ITER_TAG: str(iteration)})


def load_training_state(run_id: str) -> tuple[int, dict]:
    """:return: The last iteration of the stored state and the state"""
    name = mlflow.MlflowClient().get_run(run_id).data.tags[RESUME_ARTIFACT_TAG]
    with tempfile.TemporaryDirectory() as directory:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name, dst_path=directory)
        state = torch.load(path, weights_only=True)
    return state.pop("iteration"), state
