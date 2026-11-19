"""
Resuming interrupted training runs. Every `resume_frequency` iterations, a run stores the full state of training: the
agent with its optimizer and its experience, e.g. the replay, the curriculum and the progress counters, so that a
resumed run continues as if it had not been interrupted. Runs are tagged with a hash of their config, so that starting
training with the same config again can offer to resume them, also with more iterations to continue a finished run.
"""

import datetime
import hashlib
import json
import logging
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import torch
from mlflow.store.tracking.dbmodels.models import SqlLatestMetric, SqlMetric
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._tracking_service.utils import _get_store
from omegaconf import DictConfig, OmegaConf

from agents import CHECKPOINT_PREFIX

log = logging.getLogger(__name__)

# Settings that do not change how the agent is trained. Schedules over all iterations still change the hash, e.g. a
# learning rate decay over num_iters.
NON_TRAINING_KEYS = (
    "num_iters",
    "tags",
    "render",
    "resume",
    "test_run_id",
    "test_iter",
    "tracking_uri",
    "save_frequency",
    "resume_frequency",
)
CONFIG_HASH_TAG = "config_hash"
RESUME_ITER_TAG = "resume_iter"
RESUME_ARTIFACT_TAG = "resume_artifact"
# Tags naming the best checkpoint by the curriculum's score with that score (see train.Checkpoints), and the one with
# the most progress in greedy evaluation (see greedy_progress.py)
BEST_CHECKPOINT_ITER_TAG, BEST_SCORE_TAG = "best_checkpoint_iter", "best_score"
BEST_GREEDY_ITER_TAG = "best_greedy_iter"
# The state alternates between two files, so that an interrupted save leaves the previous state intact
RESUME_ARTIFACTS = ("resume_state_a.pt", "resume_state_b.pt")


def config_hash(cfg: DictConfig) -> str:
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    relevant = {key: value for key, value in container.items() if key not in NON_TRAINING_KEYS}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class PreviousRun:
    run_id: str
    run_name: str | None
    start_time: datetime.datetime
    # The last iteration whose state was saved
    iteration: int
    num_iters: int

    @property
    def finished(self) -> bool:
        return self.iteration >= self.num_iters


def find_previous_run(experiment_id: str, cfg_hash: str, num_iters: int) -> PreviousRun | None:
    """The most recently started run with the given config hash that can be continued, finished or not."""
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.{CONFIG_HASH_TAG} = '{cfg_hash}'",
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    for run in runs:
        tags = run.data.tags
        # Runs stopped on purpose, e.g. pruned trials of a hyperparameter search, are not resumed
        if RESUME_ITER_TAG in tags and "stopped_early" not in tags:
            start_time = datetime.datetime.fromtimestamp(run.info.start_time / 1000)
            return PreviousRun(run.info.run_id, run.info.run_name, start_time, int(tags[RESUME_ITER_TAG]), num_iters)
    return None


def decide_resume(run: PreviousRun | None, resume: bool | None, ask: Callable[[str], str] = input) -> str | None:
    """
    :param resume: Whether to resume an interrupted run, None to ask. Resuming a run that already reached num_iters
        exits instead, since there is nothing left to train.
    :return: The id of the run to resume, None to start a new one
    """
    if run is None:
        if resume:
            log.info("No interrupted run with this config, starting a new one.")
        return None
    description = (
        f"Run {run.run_name} ({run.run_id}) with the same config, started {run.start_time:%Y-%m-%d %H:%M}, "
        f"stopped after iteration {run.iteration} of {run.num_iters}."
    )
    if run.finished:
        if resume:
            raise SystemExit(f"{description} Nothing to resume, raise num_iters to continue it.")
        return None
    if resume is not None:
        return run.run_id if resume else None

    if not sys.stdin.isatty():
        raise SystemExit(f"{description} Set resume=true to continue it or resume=false to start a new run.")
    while (answer := ask(f"{description}\nResume it? [y/n] ").strip().lower()) not in ("y", "n"):
        pass
    return run.run_id if answer == "y" else None


def local_artifacts(run_id: str) -> Path:
    """
    The directory of the run's artifacts. The state of training takes several GB, so it is written and read there
    directly rather than through a copy in a temporary directory, which may be in memory.
    """
    uri = urlparse(str(mlflow.MlflowClient().get_run(run_id).info.artifact_uri))
    assert uri.scheme in ("", "file"), "Resuming needs artifacts stored locally."
    return Path(uri.path)


def save_training_state(iteration: int, state: dict) -> None:
    """Store the state of training after the given iteration in the active run."""
    active_run = mlflow.active_run()
    assert active_run is not None
    run_id = active_run.info.run_id
    previous = mlflow.MlflowClient().get_run(run_id).data.tags.get(RESUME_ARTIFACT_TAG)
    name = RESUME_ARTIFACTS[1] if previous == RESUME_ARTIFACTS[0] else RESUME_ARTIFACTS[0]
    directory = local_artifacts(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": iteration, **state}, directory / name)
    mlflow.set_tags({RESUME_ARTIFACT_TAG: name, RESUME_ITER_TAG: str(iteration)})


def load_training_state(run_id: str) -> tuple[int, dict]:
    """:return: The last iteration of the stored state and the state, whose tensors are mapped from the file"""
    name = mlflow.MlflowClient().get_run(run_id).data.tags[RESUME_ARTIFACT_TAG]
    state = torch.load(local_artifacts(run_id) / name, weights_only=True, mmap=True)
    return state.pop("iteration"), state


def discard_after(run_id: str, iteration: int) -> None:
    """
    Remove what the run logged after the given iteration, the one its state was restored from: the checkpoints, the
    metrics, also those of their greedy evaluation, and the tags naming the best of them. They belong to a course of
    training that no longer exists, since the resumed run repeats these iterations. The metrics are deleted from the
    tracking database, as MLflow has no API for it.
    """
    pattern = re.compile(re.escape(CHECKPOINT_PREFIX) + r"(\d+)")
    directory = local_artifacts(run_id)
    checkpoints = [
        path
        for path in directory.glob(f"{CHECKPOINT_PREFIX}*")
        if (match := pattern.fullmatch(path.name)) and int(match.group(1)) > iteration
    ]
    for path in checkpoints:
        shutil.rmtree(path)

    store = _get_store()
    assert isinstance(store, SqlAlchemyStore), "Discarding metrics needs a tracking database."
    with store.ManagedSessionMaker(read_only=False) as session:
        later = session.query(SqlMetric).filter(SqlMetric.run_uuid == run_id, SqlMetric.step > iteration)
        keys = [key for (key,) in later.with_entities(SqlMetric.key).distinct()]
        num_metrics = later.delete(synchronize_session=False)
        # The latest value of each metric, as MLflow keeps it for the run's overview
        session.query(SqlLatestMetric).filter(SqlLatestMetric.run_uuid == run_id, SqlLatestMetric.key.in_(keys)).delete(
            synchronize_session=False
        )
        for key in keys:
            latest = (
                session.query(SqlMetric)
                .filter(SqlMetric.run_uuid == run_id, SqlMetric.key == key)
                .order_by(SqlMetric.step.desc(), SqlMetric.timestamp.desc(), SqlMetric.value.desc())
                .first()
            )
            if latest is not None:
                session.add(
                    SqlLatestMetric(
                        run_uuid=run_id,
                        key=key,
                        value=latest.value,
                        timestamp=latest.timestamp,
                        step=latest.step,
                        is_nan=latest.is_nan,
                    )
                )

    client = mlflow.MlflowClient()
    tags = client.get_run(run_id).data.tags
    if int(tags.get(BEST_CHECKPOINT_ITER_TAG, 0)) > iteration:
        client.delete_tag(run_id, BEST_CHECKPOINT_ITER_TAG)
        client.delete_tag(run_id, BEST_SCORE_TAG)
    if int(tags.get(BEST_GREEDY_ITER_TAG, 0)) > iteration:
        client.delete_tag(run_id, BEST_GREEDY_ITER_TAG)
    if checkpoints or num_metrics:
        log.info(
            f"Discarded {len(checkpoints)} checkpoints and {num_metrics} metric values after iteration {iteration}."
        )
