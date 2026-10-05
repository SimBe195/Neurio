import numpy as np
import pytest
import torch

from agents.self_imitation import SelfImitationMemory
from test_ppo_agent import ENV_INFO, _make_agent, _outcome, _random_states

STATE_SHAPE = (ENV_INFO.total_channel_dim, ENV_INFO.height, ENV_INFO.width)


def _play(memory: SelfImitationMemory, rewards_per_step, dones_per_step, levels, actions=1):
    workers = len(rewards_per_step[0])
    for rewards, dones in zip(rewards_per_step, dones_per_step, strict=True):
        memory.record_actions(
            torch.full((workers, *STATE_SHAPE), 7.0),
            torch.zeros(workers, dtype=torch.int64),
            torch.full((workers,), actions, dtype=torch.int64),
        )
        memory.record_rewards(torch.tensor(rewards, dtype=torch.float32), dones, levels)


def test_episode_returns_and_storage():
    memory = SelfImitationMemory(num_workers=2, gamma=0.5, episodes_per_level=4)
    # Worker 0 finishes an episode of 3 steps, worker 1 is still playing
    _play(memory, [[1.0, 5.0], [2.0, 5.0], [4.0, 5.0]], [[False, False], [False, False], [True, False]], ["1-3", "1-3"])

    (episode,) = memory.episodes["1-3"]
    assert episode.total_reward == 7.0
    assert episode.returns.tolist() == pytest.approx([1 + 0.5 * 2 + 0.25 * 4, 2 + 0.5 * 4, 4.0])
    assert episode.states.dtype == torch.uint8 and episode.states.shape == (3, *STATE_SHAPE)
    assert memory.get_metrics()["sil_stored_episodes"] == 1


def test_keeps_best_episodes_per_level():
    memory = SelfImitationMemory(num_workers=1, gamma=0.9, episodes_per_level=2)
    for total, level in [(1.0, "1-3"), (3.0, "1-3"), (2.0, "1-3"), (0.5, "1-3"), (0.1, "4-4")]:
        _play(memory, [[total]], [[True]], [level])
    assert sorted(e.total_reward for e in memory.episodes["1-3"]) == [2.0, 3.0]
    # Another level has its own store, even with a much lower reward
    assert [e.total_reward for e in memory.episodes["4-4"]] == [pytest.approx(0.1)]


def test_sampling():
    memory = SelfImitationMemory(num_workers=1, gamma=0.9, episodes_per_level=2)
    assert memory.sample(8) is None
    _play(memory, [[1.0], [1.0]], [[False], [True]], ["1-1"], actions=3)
    states, _prev_actions, actions, returns = memory.sample(8)
    assert states.shape == (8, *STATE_SHAPE) and states.dtype == torch.uint8
    assert actions.tolist() == [3] * 8
    # The two steps have returns 1 + 0.9 * 1 and 1
    assert all(r == pytest.approx(1.9) or r == pytest.approx(1.0) for r in returns.tolist())


@pytest.fixture
def sil_agent(monkeypatch):
    return _make_agent(monkeypatch, sil_loss_weight=1.0, sil_updates=20, sil_batch_size=32)


def test_agent_records_its_episodes(sil_agent):
    rng = np.random.default_rng(12)
    for t in range(3):
        sil_agent.feed_observation(_random_states(rng))
        sil_agent.next_actions()
        sil_agent.give_reward(
            _outcome([1.0, 2.0], [t == 2, False], [False, False], bonus=[9.0, 9.0], levels=["1-3"] * 2)
        )
    (episode,) = sil_agent.self_imitation.episodes["1-3"]
    # The game's reward only, without the bonus
    assert episode.total_reward == 3.0
    assert len(episode.actions) == 3


def test_update_imitates_episodes_better_than_expected(sil_agent):
    memory = sil_agent.self_imitation
    state = sil_agent._state_to_tensor(_random_states(np.random.default_rng(13)))[:1]
    memory._trajectories = memory._trajectories[:1]

    def log_prob(action):
        with torch.no_grad():
            logits, _ = sil_agent.actor_critic(state, torch.zeros(1, dtype=torch.int64))
        return torch.log_softmax(logits, -1)[0, action].item()

    def store(action, reward):
        memory.record_actions(state, torch.zeros(1, dtype=torch.int64), torch.tensor([action]))
        memory.record_rewards(torch.tensor([reward]), [True], ["1-3"])

    # A return far above the value estimate is imitated
    store(action=2, reward=100.0)
    before = log_prob(2)
    metrics = sil_agent._self_imitation_update()
    assert log_prob(2) > before + 0.5
    assert metrics["sil_positive_share"] == 1.0

    # A return far below the value estimate is ignored
    memory.episodes.clear()
    memory._samples = None
    store(action=1, reward=-100.0)
    before = log_prob(1)
    metrics = sil_agent._self_imitation_update()
    assert log_prob(1) == pytest.approx(before, abs=1e-6)
    assert metrics["sil_positive_share"] == 0.0


def test_disabled_with_zero_weight(monkeypatch):
    assert _make_agent(monkeypatch, sil_loss_weight=0.0).self_imitation is None
