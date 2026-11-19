import datetime

import mlflow
import pytest
import torch

from agents import checkpoint_name
from resume import (
    BEST_CHECKPOINT_ITER_TAG,
    BEST_GREEDY_ITER_TAG,
    BEST_SCORE_TAG,
    PreviousRun,
    decide_resume,
    discard_after,
    local_artifacts,
)

RUN = PreviousRun("abc123", "brave-fox", datetime.datetime(2026, 10, 6, 12, 0), iteration=140, num_iters=1000)
FINISHED = PreviousRun("abc123", "brave-fox", datetime.datetime(2026, 10, 6, 12, 0), iteration=1000, num_iters=1000)


def test_explicit_choice():
    assert decide_resume(RUN, resume=True) == "abc123"
    assert decide_resume(RUN, resume=False) is None
    assert decide_resume(None, resume=True) is None
    assert decide_resume(None, resume=None) is None


def test_finished_run_is_not_resumed():
    with pytest.raises(SystemExit, match="Nothing to resume"):
        decide_resume(FINISHED, resume=True)
    assert decide_resume(FINISHED, resume=False) is None
    assert decide_resume(FINISHED, resume=None) is None


def test_asks_in_a_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    questions = []
    answers = iter(["maybe", " Y "])

    def ask(question):
        questions.append(question)
        return next(answers)

    assert decide_resume(RUN, resume=None, ask=ask) == "abc123"
    # Asked again after an invalid answer
    assert len(questions) == 2
    assert "iteration 140 of 1000" in questions[0]
    assert decide_resume(RUN, resume=None, ask=lambda _: "n") is None


def test_refuses_to_guess_without_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit, match="resume=true"):
        decide_resume(RUN, resume=None)


def test_restored_state_survives_safe_loading(tmp_path, monkeypatch):
    import numpy as np

    from test_ppo_agent import _make_agent, _outcome, _random_states

    agent = _make_agent(monkeypatch, sil_loss_weight=1.0)
    rng = np.random.default_rng(3)
    for t in range(3):
        agent.act(_random_states(rng))
        agent.observe(_outcome([1.0, 2.0], [t == 2, False], [False, False], levels=["1-3"] * 2))
    agent.update_step = 7
    torch.save(agent.state_dict(), tmp_path / "state.pt")

    restored = _make_agent(monkeypatch, sil_loss_weight=1.0)
    restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    assert restored.update_step == 7
    (episode,) = restored.self_imitation.episodes["1-3"]
    assert episode.total_reward == 3.0
    assert torch.equal(episode.frames, agent.self_imitation.episodes["1-3"][0].frames)
    for name, value in agent.actor_critic.state_dict().items():
        assert torch.equal(value, restored.actor_critic.state_dict()[name])


@pytest.fixture
def run_id(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    experiment_id = mlflow.create_experiment("resume", artifact_location=str(tmp_path / "artifacts"))
    with mlflow.start_run(experiment_id=experiment_id) as run:
        for step in range(1, 6):
            mlflow.log_metrics({"loss": 10.0 - step, "levels/success_1-1": step / 10}, step=step)
        mlflow.log_metric("greedy/progress", 3000.0, step=4)
        mlflow.set_tags({BEST_CHECKPOINT_ITER_TAG: "4", BEST_SCORE_TAG: "1.5", BEST_GREEDY_ITER_TAG: "2"})
    for iteration in (2, 4, 5):
        (local_artifacts(run.info.run_id) / checkpoint_name(iteration)).mkdir(parents=True)
    return run.info.run_id


def test_discards_what_was_logged_after_the_restored_state(run_id):
    discard_after(run_id, 3)

    client = mlflow.MlflowClient()
    assert [(m.step, m.value) for m in client.get_metric_history(run_id, "loss")] == [(1, 9.0), (2, 8.0), (3, 7.0)]
    assert client.get_metric_history(run_id, "greedy/progress") == []
    run = client.get_run(run_id)
    # The latest values in the run's overview are those of the restored iteration
    assert run.data.metrics == {"loss": 7.0, "levels/success_1-1": 0.3}
    assert sorted(path.name for path in local_artifacts(run_id).iterdir()) == [checkpoint_name(2)]
    # Only tags naming discarded checkpoints are removed
    assert BEST_CHECKPOINT_ITER_TAG not in run.data.tags and BEST_SCORE_TAG not in run.data.tags
    assert run.data.tags[BEST_GREEDY_ITER_TAG] == "2"


def test_nothing_to_discard(run_id):
    discard_after(run_id, 5)
    assert len(mlflow.MlflowClient().get_metric_history(run_id, "loss")) == 5
    assert len(list(local_artifacts(run_id).iterdir())) == 3
