import datetime

import pytest
import torch

from resume import ResumableRun, decide_resume

RUN = ResumableRun("abc123", "brave-fox", datetime.datetime(2026, 10, 6, 12, 0), iteration=140, num_iters=1000)


def test_explicit_choice():
    assert decide_resume(RUN, resume=True) == "abc123"
    assert decide_resume(RUN, resume=False) is None
    assert decide_resume(None, resume=True) is None
    assert decide_resume(None, resume=None) is None


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
        agent.feed_observation(_random_states(rng))
        agent.next_actions()
        agent.give_reward(_outcome([1.0, 2.0], [t == 2, False], [False, False], levels=["1-3"] * 2))
    agent.update_step = 7
    torch.save(agent.state_dict(), tmp_path / "state.pt")

    restored = _make_agent(monkeypatch, sil_loss_weight=1.0)
    restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    assert restored.update_step == 7
    (episode,) = restored.self_imitation.episodes["1-3 small"]
    assert episode.total_reward == 3.0
    assert torch.equal(episode.frames, agent.self_imitation.episodes["1-3 small"][0].frames)
    for name, value in agent.actor_critic.state_dict().items():
        assert torch.equal(value, restored.actor_critic.state_dict()[name])
