import pytest
import torch
from agents.advantage import GaeEstimator


def test_output_shapes():
    gae = GaeEstimator(0.99, 0.95)

    rewards = torch.randn(5, dtype=torch.float32)
    values = torch.randn(6, dtype=torch.float32)
    dones = torch.randint(0, 1, (5,), dtype=torch.float32)

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape


def test_output_shapes_multi_worker():
    gae = GaeEstimator(0.99, 0.95)

    rewards = torch.randn((5, 3), dtype=torch.float32)
    values = torch.randn((6, 3), dtype=torch.float32)
    dones = torch.randint(0, 1, (5, 3), dtype=torch.float32)

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape


def test_shape_mismatch():
    gae = GaeEstimator(0.99, 0.95)

    rewards = torch.tensor([1, 0, 1], dtype=torch.float32)
    values = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    dones = torch.tensor([0, 1], dtype=torch.float32)

    with pytest.raises(ValueError):
        gae.get_advantage_returns(rewards, values, dones)


def test_batch_dimension_mismatch():
    gae = GaeEstimator(0.99, 0.95)

    rewards = torch.randn(5, 2)
    values = torch.randn(6, 2)
    dones = torch.randn(5, 3)

    with pytest.raises(ValueError):
        gae.get_advantage_returns(rewards, values, dones)


def test_dimensionality_check():
    gae = GaeEstimator(0.99, 0.95)

    rewards = torch.randn(5, 3, 2)
    values = torch.randn(6, 3, 2)
    dones = torch.randn(5, 3, 2)

    with pytest.raises(ValueError):
        gae.get_advantage_returns(rewards, values, dones)


def test_gae_calculation_single_step():
    gamma = 0.99
    tau = 0.95
    gae = GaeEstimator(gamma, tau)

    rewards = torch.randn(1, dtype=torch.float32)
    values = torch.randn(2, dtype=torch.float32)
    dones = torch.tensor([0], dtype=torch.float32)

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    expected_advantages = torch.tensor([rewards[0] + gamma * values[1] - values[0]], dtype=torch.float32)
    expected_returns = torch.tensor([rewards[0] + gamma * values[1]], dtype=torch.float32)

    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_returns)


def test_gae_calculation_single_step_done():
    gamma = 0.99
    tau = 0.95
    gae = GaeEstimator(gamma, tau)

    rewards = torch.randn(1, dtype=torch.float32)
    values = torch.randn(2, dtype=torch.float32)
    dones = torch.tensor([1], dtype=torch.float32)

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    expected_advantages = torch.tensor([rewards[0] - values[0]], dtype=torch.float32)
    expected_returns = torch.tensor([rewards[0]], dtype=torch.float32)

    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_returns)


def test_gae_calculation_multi_step():
    gamma = 0.99
    tau = 0.95

    gae = GaeEstimator(gamma, tau)

    rewards = torch.randn(3, dtype=torch.float32)
    values = torch.randn(4, dtype=torch.float32)
    dones = torch.tensor([0, 0, 0], dtype=torch.float32)

    delta_0 = rewards[0] + gamma * values[1] - values[0]
    delta_1 = rewards[1] + gamma * values[2] - values[1]
    delta_2 = rewards[2] + gamma * values[3] - values[2]

    gae_0 = delta_0 + (gamma * tau) * delta_1 + (gamma * tau) ** 2 * delta_2
    return_0 = gae_0 + values[0]
    gae_1 = delta_1 + (gamma * tau) * delta_2
    return_1 = gae_1 + values[1]
    gae_2 = delta_2
    return_2 = gae_2 + values[2]

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    # Manually calculated expected values
    expected_advantages = torch.tensor([gae_0, gae_1, gae_2], dtype=torch.float32)
    expected_returns = torch.tensor([return_0, return_1, return_2], dtype=torch.float32)

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)
    assert torch.allclose(returns, expected_returns, atol=1e-4)


def test_gae_calculation_multi_step_done_end():
    gamma = 0.99
    tau = 0.95

    gae = GaeEstimator(gamma, tau)

    rewards = torch.randn(3, dtype=torch.float32)
    values = torch.randn(4, dtype=torch.float32)
    dones = torch.tensor([0, 0, 1], dtype=torch.float32)

    delta_0 = rewards[0] + gamma * values[1] - values[0]
    delta_1 = rewards[1] + gamma * values[2] - values[1]
    delta_2 = rewards[2] - values[2]

    gae_0 = delta_0 + (gamma * tau) * delta_1 + (gamma * tau) ** 2 * delta_2
    return_0 = gae_0 + values[0]
    gae_1 = delta_1 + (gamma * tau) * delta_2
    return_1 = gae_1 + values[1]
    gae_2 = delta_2
    return_2 = gae_2 + values[2]

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    # Manually calculated expected values
    expected_advantages = torch.tensor([gae_0, gae_1, gae_2], dtype=torch.float32)
    expected_returns = torch.tensor([return_0, return_1, return_2], dtype=torch.float32)

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)
    assert torch.allclose(returns, expected_returns, atol=1e-4)


def test_gae_calculation_multi_step_done_middle():
    gamma = 0.99
    tau = 0.95

    gae = GaeEstimator(gamma, tau)

    rewards = torch.randn(3, dtype=torch.float32)
    values = torch.randn(4, dtype=torch.float32)
    dones = torch.tensor([0, 1, 0], dtype=torch.float32)

    delta_0 = rewards[0] + gamma * values[1] - values[0]
    delta_1 = rewards[1] - values[1]
    delta_2 = rewards[2] + gamma * values[3] - values[2]

    gae_0 = delta_0 + (gamma * tau) * delta_1
    return_0 = gae_0 + values[0]
    gae_1 = delta_1
    return_1 = gae_1 + values[1]
    gae_2 = delta_2
    return_2 = gae_2 + values[2]

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    # Manually calculated expected values
    expected_advantages = torch.tensor([gae_0, gae_1, gae_2], dtype=torch.float32)
    expected_returns = torch.tensor([return_0, return_1, return_2], dtype=torch.float32)

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)
    assert torch.allclose(returns, expected_returns, atol=1e-4)


def test_gae_calculation_single_step_multi_worker():
    gamma = 0.99
    tau = 0.95
    gae = GaeEstimator(gamma, tau)

    rewards = torch.randn((1, 2), dtype=torch.float32)
    values = torch.randn((2, 2), dtype=torch.float32)
    dones = torch.tensor([[0, 0]], dtype=torch.float32)

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    expected_advantages = torch.tensor(
        [[rewards[0][0] + gamma * values[1][0] - values[0][0], rewards[0][1] + gamma * values[1][1] - values[0][1]]],
        dtype=torch.float32,
    )
    expected_returns = torch.tensor(
        [[rewards[0][0] + gamma * values[1][0], rewards[0][1] + gamma * values[1][1]]], dtype=torch.float32
    )

    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_returns)


def test_gae_calculation_single_step_done_multi_worker():
    gamma = 0.99
    tau = 0.95
    gae = GaeEstimator(gamma, tau)

    rewards = torch.randn((1, 2), dtype=torch.float32)
    values = torch.randn((2, 2), dtype=torch.float32)
    dones = torch.tensor([[0, 1]], dtype=torch.float32)

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    expected_advantages = torch.tensor(
        [[rewards[0][0] + gamma * values[1][0] - values[0][0], rewards[0][1] - values[0][1]]],
        dtype=torch.float32,
    )
    expected_returns = torch.tensor([[rewards[0][0] + gamma * values[1][0], rewards[0][1]]], dtype=torch.float32)

    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(returns, expected_returns)


def test_gae_calculation_multi_step_multi_worker():
    gamma = 0.99
    tau = 0.95

    gae = GaeEstimator(gamma, tau)

    rewards = torch.randn((3, 2), dtype=torch.float32)
    values = torch.randn((4, 2), dtype=torch.float32)
    dones = torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.float32)

    delta_0_0 = rewards[0][0] + gamma * values[1][0] - values[0][0]
    delta_1_0 = rewards[1][0] + gamma * values[2][0] - values[1][0]
    delta_2_0 = rewards[2][0] + gamma * values[3][0] - values[2][0]

    delta_0_1 = rewards[0][1] + gamma * values[1][1] - values[0][1]
    delta_1_1 = rewards[1][1] + gamma * values[2][1] - values[1][1]
    delta_2_1 = rewards[2][1] + gamma * values[3][1] - values[2][1]

    gae_0_0 = delta_0_0 + (gamma * tau) * delta_1_0 + (gamma * tau) ** 2 * delta_2_0
    return_0_0 = gae_0_0 + values[0][0]
    gae_1_0 = delta_1_0 + (gamma * tau) * delta_2_0
    return_1_0 = gae_1_0 + values[1][0]
    gae_2_0 = delta_2_0
    return_2_0 = gae_2_0 + values[2][0]

    gae_0_1 = delta_0_1 + (gamma * tau) * delta_1_1 + (gamma * tau) ** 2 * delta_2_1
    return_0_1 = gae_0_1 + values[0][1]
    gae_1_1 = delta_1_1 + (gamma * tau) * delta_2_1
    return_1_1 = gae_1_1 + values[1][1]
    gae_2_1 = delta_2_1
    return_2_1 = gae_2_1 + values[2][1]

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    # Manually calculated expected values
    expected_advantages = torch.tensor(
        [[gae_0_0, gae_0_1], [gae_1_0, gae_1_1], [gae_2_0, gae_2_1]], dtype=torch.float32
    )
    expected_returns = torch.tensor(
        [[return_0_0, return_0_1], [return_1_0, return_1_1], [return_2_0, return_2_1]], dtype=torch.float32
    )

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)
    assert torch.allclose(returns, expected_returns, atol=1e-4)


def test_gae_calculation_multi_step_done_end_multi_worker():
    gamma = 0.99
    tau = 0.95

    gae = GaeEstimator(gamma, tau)

    rewards = torch.randn((3, 2), dtype=torch.float32)
    values = torch.randn((4, 2), dtype=torch.float32)
    dones = torch.tensor([[0, 0], [0, 0], [1, 0]], dtype=torch.float32)

    delta_0_0 = rewards[0][0] + gamma * values[1][0] - values[0][0]
    delta_1_0 = rewards[1][0] + gamma * values[2][0] - values[1][0]
    delta_2_0 = rewards[2][0] - values[2][0]

    delta_0_1 = rewards[0][1] + gamma * values[1][1] - values[0][1]
    delta_1_1 = rewards[1][1] + gamma * values[2][1] - values[1][1]
    delta_2_1 = rewards[2][1] + gamma * values[3][1] - values[2][1]

    gae_0_0 = delta_0_0 + (gamma * tau) * delta_1_0 + (gamma * tau) ** 2 * delta_2_0
    return_0_0 = gae_0_0 + values[0][0]
    gae_1_0 = delta_1_0 + (gamma * tau) * delta_2_0
    return_1_0 = gae_1_0 + values[1][0]
    gae_2_0 = delta_2_0
    return_2_0 = gae_2_0 + values[2][0]

    gae_0_1 = delta_0_1 + (gamma * tau) * delta_1_1 + (gamma * tau) ** 2 * delta_2_1
    return_0_1 = gae_0_1 + values[0][1]
    gae_1_1 = delta_1_1 + (gamma * tau) * delta_2_1
    return_1_1 = gae_1_1 + values[1][1]
    gae_2_1 = delta_2_1
    return_2_1 = gae_2_1 + values[2][1]

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    # Manually calculated expected values
    expected_advantages = torch.tensor(
        [[gae_0_0, gae_0_1], [gae_1_0, gae_1_1], [gae_2_0, gae_2_1]], dtype=torch.float32
    )
    expected_returns = torch.tensor(
        [[return_0_0, return_0_1], [return_1_0, return_1_1], [return_2_0, return_2_1]], dtype=torch.float32
    )

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)
    assert torch.allclose(returns, expected_returns, atol=1e-4)


def test_gae_calculation_multi_step_done_middle_multi_worker():
    gamma = 0.99
    tau = 0.95

    gae = GaeEstimator(gamma, tau)

    rewards = torch.randn((3, 2), dtype=torch.float32)
    values = torch.randn((4, 2), dtype=torch.float32)
    dones = torch.tensor([[0, 0], [1, 0], [0, 0]], dtype=torch.float32)

    delta_0_0 = rewards[0][0] + gamma * values[1][0] - values[0][0]
    delta_1_0 = rewards[1][0] - values[1][0]
    delta_2_0 = rewards[2][0] + gamma * values[3][0] - values[2][0]

    delta_0_1 = rewards[0][1] + gamma * values[1][1] - values[0][1]
    delta_1_1 = rewards[1][1] + gamma * values[2][1] - values[1][1]
    delta_2_1 = rewards[2][1] + gamma * values[3][1] - values[2][1]

    gae_0_0 = delta_0_0 + (gamma * tau) * delta_1_0
    return_0_0 = gae_0_0 + values[0][0]
    gae_1_0 = delta_1_0
    return_1_0 = gae_1_0 + values[1][0]
    gae_2_0 = delta_2_0
    return_2_0 = gae_2_0 + values[2][0]

    gae_0_1 = delta_0_1 + (gamma * tau) * delta_1_1 + (gamma * tau) ** 2 * delta_2_1
    return_0_1 = gae_0_1 + values[0][1]
    gae_1_1 = delta_1_1 + (gamma * tau) * delta_2_1
    return_1_1 = gae_1_1 + values[1][1]
    gae_2_1 = delta_2_1
    return_2_1 = gae_2_1 + values[2][1]

    advantages, returns = gae.get_advantage_returns(rewards, values, dones)

    # Manually calculated expected values
    expected_advantages = torch.tensor(
        [[gae_0_0, gae_0_1], [gae_1_0, gae_1_1], [gae_2_0, gae_2_1]], dtype=torch.float32
    )
    expected_returns = torch.tensor(
        [[return_0_0, return_0_1], [return_1_0, return_1_1], [return_2_0, return_2_1]], dtype=torch.float32
    )

    assert torch.allclose(advantages, expected_advantages, atol=1e-4)
    assert torch.allclose(returns, expected_returns, atol=1e-4)
