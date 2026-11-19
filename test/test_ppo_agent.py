import itertools

import numpy as np
import pytest
import torch

from agents import PPOAgent, StepOutcome
from agents.ppo.advantage import GaeEstimator
from config.agent import PPOAgentConfig
from config.learning_rate import ConstantLRConfig
from config.model import NatureActorCriticConfig
from environment import EnvironmentInfo

ENV_INFO = EnvironmentInfo(width=12, height=10, stack_frames=2, image_channels=1, num_actions=3, num_workers=2)


@pytest.fixture
def agent(monkeypatch) -> PPOAgent:
    return _make_agent(monkeypatch)


def _make_agent(monkeypatch, **overrides) -> PPOAgent:
    # Self-imitation is on by default, the tests enable it where they need it
    overrides = {"sil_loss_weight": 0.0, **overrides}
    # Also covers construction on machines without a GPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    config = PPOAgentConfig(
        model=NatureActorCriticConfig(
            action_embedding_size=4, num_filters=[4, 4], kernel_sizes=[3, 3], strides=[2, 1], fc_size=8
        ),
        learning_rate=ConstantLRConfig(learning_rate=1e-2),
        num_stack_frames=ENV_INFO.stack_frames,
        gamma=0.9,
        tau=0.8,
        epochs_per_update=2,
        batch_size=4,
        clip_param=0.2,
        clip_value=True,
        critic_loss_weight=0.5,
        initial_entropy_loss_weight=0.01,
        entropy_decay_updates=10,
        final_entropy_loss_weight=0.001,
        grad_clip_norm=0.5,
        **overrides,
    )
    return PPOAgent(config, ENV_INFO)


def _random_states(rng: np.random.Generator) -> np.ndarray:
    shape = (ENV_INFO.num_workers, ENV_INFO.stack_frames, ENV_INFO.height, ENV_INFO.width, ENV_INFO.image_channels)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _outcome(rewards, terminated, truncated, final_observations=None, levels=None, observations=None) -> StepOutcome:
    return StepOutcome(
        observations if observations is not None else _random_states(np.random.default_rng()),
        np.asarray(rewards),
        np.asarray(terminated),
        np.asarray(truncated),
        levels or ["1-1"] * len(rewards),
        final_observations,
        game_rewards=np.asarray(rewards, dtype=float),
    )


def _play(agent: PPOAgent, steps: int, rng: np.random.Generator) -> np.ndarray:
    """:return: The observations after the last step"""
    states = _random_states(rng)
    for t in range(steps):
        actions = agent.act(states)
        assert len(actions) == ENV_INFO.num_workers
        final_observations = np.empty(ENV_INFO.num_workers, dtype=object)
        final_observations[:] = list(states)
        states = _random_states(rng)
        rewards = rng.normal(size=ENV_INFO.num_workers)
        agent.observe(_outcome(rewards, [t == 2, False], [False, t == 3], final_observations, observations=states))
    return states


def test_runs_on_cpu(agent):
    assert agent.device == torch.device("cpu")


def test_state_to_tensor_layout(agent):
    states = _random_states(np.random.default_rng(0))
    tensor = agent._state_to_tensor(states)

    assert tensor.shape == (ENV_INFO.num_workers, ENV_INFO.total_channel_dim, ENV_INFO.height, ENV_INFO.width)
    assert tensor.dtype == torch.float32
    for frame in range(ENV_INFO.stack_frames):
        assert np.array_equal(tensor[:, frame].numpy(), states[:, frame, :, :, 0])


def test_update_bootstraps_from_next_state(agent, monkeypatch):
    next_states = _play(agent, steps=5, rng=np.random.default_rng(0))

    with torch.no_grad():
        _, expected_next_values = agent.actor_critic(agent._state_to_tensor(next_states), agent.prev_actions)
    buffered_values = agent.experience_buffer.stack("values")

    captured = {}
    original = GaeEstimator.get_advantage_returns

    def spy(self, rewards, values, dones):
        captured["values"] = values
        return original(self, rewards, values, dones)

    monkeypatch.setattr(GaeEstimator, "get_advantage_returns", spy)
    agent.update()

    assert torch.allclose(captured["values"][:-1], buffered_values)
    assert torch.allclose(captured["values"][-1], expected_next_values)


def test_update_changes_parameters(agent):
    _play(agent, steps=6, rng=np.random.default_rng(1))
    before = [p.detach().clone() for p in agent.actor_critic.parameters()]

    metrics = agent.update()

    assert agent.update_step == 1
    assert 0 < metrics["behavior/entropy"] <= np.log(agent.env_info.num_actions)
    after = list(agent.actor_critic.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
    assert all(torch.isfinite(a).all() for a in after)


def test_each_update_learns_from_the_steps_since_the_previous_one(agent):
    rng = np.random.default_rng(2)
    for _ in range(3):
        _play(agent, steps=5, rng=rng)
        assert len(agent.experience_buffer) == 5
        agent.update()
        assert len(agent.experience_buffer) == 0
    assert agent.update_step == 3


def test_truncated_episodes_are_bootstrapped(agent):
    rng = np.random.default_rng(3)
    actions = agent.act(_random_states(rng))

    final_observations = np.empty(ENV_INFO.num_workers, dtype=object)
    final_observations[:] = [_random_states(rng)[0], _random_states(rng)[0]]
    with torch.no_grad():
        _, final_values = agent.actor_critic(
            agent._state_to_tensor(np.stack(final_observations)), torch.tensor(actions, dtype=torch.int64)
        )

    # Worker 0 terminated, worker 1 was truncated
    agent.observe(_outcome([1.0, 1.0], [True, False], [False, True], final_observations))

    # The value of the final state is already on the scale the critic learns
    rewards = agent.experience_buffer.stack("rewards")[-1]
    std = agent.reward_scale.std
    assert rewards[0] == pytest.approx(1.0 / std)
    assert rewards[1] == pytest.approx(1.0 / std + agent.config.gamma * final_values[1].item(), rel=1e-5)
    assert agent.experience_buffer.stack("dones")[-1].tolist() == [1, 1]
    # The next episodes start without a previous action
    assert agent.prev_actions.tolist() == [0, 0]


def test_no_bootstrap_without_truncation(agent):
    agent.act(_random_states(np.random.default_rng(4)))
    agent.observe(_outcome([1.0, 2.0], [True, False], [False, False]))
    rewards = agent.experience_buffer.stack("rewards")[-1] * agent.reward_scale.std
    assert rewards.tolist() == pytest.approx([1.0, 2.0])


def test_greedy_actions_are_most_likely_and_not_learned_from(agent):
    states = _random_states(np.random.default_rng(5))
    with torch.no_grad():
        logits, _ = agent.actor_critic(agent._state_to_tensor(states), agent.prev_actions)
    assert agent.act(states, greedy=True) == logits.argmax(-1).tolist()
    agent.observe(_outcome([1.0, 2.0], [False, False], [False, False]))
    assert len(agent.experience_buffer) == 0


def test_entropy_weight_schedule(agent):
    weights = []
    decay_updates = agent.config.entropy_decay_updates
    for step in range(decay_updates + 3):
        agent.update_step = step
        weights.append(agent._current_entropy_loss_weight())

    assert weights[0] == pytest.approx(0.01)
    assert weights[decay_updates // 2] == pytest.approx(0.0055)
    assert weights[decay_updates] == pytest.approx(0.001)
    # Then stays at the final weight
    assert weights[-1] == pytest.approx(0.001)
    assert all(a >= b for a, b in itertools.pairwise(weights))


@pytest.mark.parametrize("normalize", [True, False])
def test_advantage_normalization(agent, normalize):
    agent.config.normalize_advantages = normalize
    logits = torch.zeros(4, ENV_INFO.num_actions)
    actions = torch.zeros(4, dtype=torch.int64)
    old_log_probs = torch.full((4,), -float(np.log(ENV_INFO.num_actions)))
    advantages = torch.tensor([1.0, 2.0, 3.0, 4.0]) * 100

    # With ratio 1, the actor loss is the negative mean advantage
    loss, _ = agent._actor_loss(logits, actions, old_log_probs, advantages)
    assert loss.item() == pytest.approx(0.0 if normalize else -250.0, abs=1e-5)


def test_network_sees_the_previous_action_of_the_episode(agent):
    rng = np.random.default_rng(8)
    first = agent.act(_random_states(rng))
    # Worker 0's episode ends
    agent.observe(_outcome([0.0, 0.0], [True, False], [False, False]))
    second = agent.act(_random_states(rng))
    agent.observe(_outcome([0.0, 0.0], [False, False], [False, False]))

    assert agent.experience_buffer.stack("prev_actions").tolist() == [[0, 0], [0, first[1]]]
    assert agent.experience_buffer.stack("actions").tolist() == [first, second]
